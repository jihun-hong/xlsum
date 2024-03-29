import os
import glob
import time
import torch
import random
import signal
import distributed

from others.utils import init_logger, logger
from models import builder
from models.builder import Summarizer
from models.trainer import build_trainer
from models.loader import Dataloader, load_dataset

from pytorch_transformers import XLNetConfig

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and
    propagates the traceback to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args, device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):

    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


def train(args, device_id):
    # Start logger.
    init_logger(args.log_file)

    # Configure training device.
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    # Configure manual seed.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set CUDA device.
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Configure manual seed.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Dataloader used for training.
    def train_iter_fct():
        return Dataloader(args, load_dataset(args, 'train', shuffle=True),
                          args.batch_size, device, shuffle=True, is_test=False)

    # Build the model.
    model = Summarizer(args, device, load_pretrained=True)

    # Configure the checkpoint.
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)

        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])

        model.load_cp(checkpoint)
        optim = builder.build_optim(args, model, checkpoint)
    else:
        optim = builder.build_optim(args, model, None)
    logger.info(model)

    # Train the model
    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)


def test(args, device_id, pt, step):

    # Configure device name.
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    # Configure model checkpoint.
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    # Load XLNet configuration.
    config = XLNetConfig.from_pretrained(args.config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter = Dataloader(args, load_dataset(args, 'test', shuffle=False),
                           args.batch_size, device, shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)


def wait_and_validate(args, device_id):

    timestep = 0
    if args.test_all:
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args,  device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args,  device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,  device_id, cp, step)
                    test(args,  device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args,  device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    config = XLNetConfig.from_pretrained(args.config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter = Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                    args.batch_size, device, shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()
