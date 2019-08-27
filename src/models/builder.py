import torch
import torch.nn as nn
from pytorch_transformers import XLNetModel
from torch.nn.init import xavier_uniform_

from models.layer import Classifier
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class XLNet(nn.Module):
    def __init__(self, temp_dir, load_pretrained, xlnet_config=None):
        super(XLNet, self).__init__()
        if load_pretrained:
            self.model = XLNetModel.from_pretrained('xlnet-base-cased', cache_dir=temp_dir)
        else:
            self.model = XLNetModel(xlnet_config)

    def forward(self, x, mask):
        output = self.model(x, attention_mask=mask)
        top_vec = output[0]
        return top_vec


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained=True, config=None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.xlnet = XLNet(args.temp_dir, load_pretrained, xlnet_config=config)
        self.layer = Classifier(self.xlnet.model.config.d_model)

        # Initialize parameters.
        if args.param_init != 0.0:
            for p in self.layer.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.layer.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, clss, mask, mask_cls):
        top_vec = self.xlnet(x, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
