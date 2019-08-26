import os
import gc
import glob
import json
import torch
import urllib
import zipfile
import subprocess

from os.path import join
from pathlib import Path
from multiprocess import Pool
from pytorch_transformers import XLNetTokenizer

from prepro.utils import hashhex, clean
from prepro.utils import greedy, combination
from others.utils import logger


# Configure Stanford CoreNLP.
def configure_tokenizer(args):
    link_map = {'stanford-corenlp': 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'}

    def download_and_extract(link, dir):
        print("Downloading and extracting {}...".format(link))
        data_file = "{}.zip".format(link)
        urllib.request.urlretrieve(link_map[link], data_file)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(dir)
        os.remove(data_file)
        print("\tCompleted!")

    if not os.path.isdir(args.tokenizer_dir):
        os.mkdir(args.tokenizer_dir)
        download_and_extract('stanford-corenlp', args.tokenizer_dir)

    os.environ["CLASSPATH"] = "{}stanford-corenlp-full-{}/stanford-corenlp-{}.jar".format(args.tokenizer_dir,
                                                                                          args.tokenizer_date,
                                                                                          args.tokenizer_ver)


# Tokenize the raw stories using Stanford CoreNLP.
def tokenize(args):
    # Directory to original story files.
    stories_dir = os.path.abspath(args.raw_path)
    # Directory to tokenized data files.
    tokenized_dir = os.path.abspath(args.tokenized_path)

    print("Starting to tokenize %s to %s..." % (stories_dir, tokenized_dir))

    # create IO list file
    stories = os.listdir(stories_dir)
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if not s.endswith('story'):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))

    configure_tokenizer(args)

    # Tokenize using Stanford CoreNLP.
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize, ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt',
               '-outputFormat', 'json', '-outputDirectory', tokenized_dir]
    subprocess.call(command)
    os.remove("mapping_for_corenlp.txt")

    # Check if number of tokenized stories match number of original stories.
    num_original = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_dir))
    if num_original != num_tokenized:
        print("Warning! The tokenized directory %s contains %i files, but the original directory %s contains %i files"
              % (tokenized_dir, num_tokenized, stories_dir, num_original))


def _format_json(param):
    f, args = param
    print(f)
    src, tgt = load_json(f, args.lower)
    return {'src': src, 'tgt': tgt}


def format_json(args):
    corpus_mapping = {}
    type_list = ['valid', 'test', 'train']

    for corpus_type in type_list:
        temp = []
        for line in open(join(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}

    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(join(args.tokenized_path, '*.json')):
        real_name = os.path.basename(f).split('.')[0]
        if real_name in corpus_mapping['valid']:
            valid_files.append(f)
        elif real_name in corpus_mapping['test']:
            test_files.append(f)
        elif real_name in corpus_mapping['train']:
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in type_list:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        data_set = []
        p_ct = 0
        for d in pool.imap_unordered(_format_json, a_lst):
            data_set.append(d)
            if len(data_set) > args.shard_size:
                pt_file = "{:s}.{:s}.{:d}.json".format(args.json_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(data_set))
                    p_ct += 1
                    data_set = []
                    print("saving ...")

        pool.close()
        pool.join()

        if len(data_set) > 0:
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(data_set))
                p_ct += 1


def load_json(p, lower):
    src = []
    tgt = []
    flag = False

    for sent in json.load(open(p, 'rt', encoding='UTF8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        # change all words to lower case
        if lower:
            tokens = [t.lower() for t in tokens]
        if tokens[0] == '@highlight':
            flag = True
            continue
        if flag:
            tgt.append(tokens)
            flag = False
        else:
            src.append(tokens)

    src = [clean(' '.join(sent)).split() for sent in src]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return src, tgt


def format_xlnet(args):
    if args.dataset is not '':
        data_type = [args.dataset]
    else:
        data_type = ['train', 'valid', 'test']

    for corpus_type in data_type:
        a_lst = []
        for json_f in glob.glob(join(args.json_path, '*' + corpus_type + '.*.json')):
            real_name = os.path.basename(json_f)
            print(real_name)
            a_lst.append((json_f, args, join(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)

        pool = Pool(args.n_cpus)
        for _ in pool.imap(_format_xlnet, a_lst):
            pass

        pool.close()
        pool.join()


def _format_xlnet(param):
    json_file, args, save_file = param

    # if file already exists, ignore
    if os.path.exists(save_file):
        logger.info('Ignore %s' % save_file)
        return

    xlnet = XLData(args)
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    data_set = []

    # iterate over text in json_file
    for d in jobs:
        # generate oracle ids
        src, tgt = d['src'], d['tgt']
        if args.oracle_mode == 'greedy':
            oracle_ids = greedy(src, tgt, 3)
        elif args.oracle_mode == 'combination':
            oracle_ids = combination(src, tgt, 3)

        # process data using oracle ids
        xl_data = xlnet.process(src, tgt, oracle_ids)
        if xl_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = xl_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        data_set.append(b_data_dict)

    # save file with torch
    logger.info('Saving to %s' % save_file)
    torch.save(data_set, save_file)
    gc.collect()


class XLData:
    def __init__(self, args):
        self.args = args
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.sep_id = self.tokenizer.sep_token
        self.cls_id = self.tokenizer.cls_token
        self.pad_id = self.tokenizer.pad_token

    def process(self, src, tgt, oracle_ids):
        """
        :param src: source document (array of sentences, each sentence is array of words)
        :param tgt: target summary (array of sentences, each sentence is array of words)
        :param oracle_ids: the ids for sentences that are included in the oracle summary
        :return src_subtoken_idxs: array of token ids for the original article
        :return labels: array that indicate which sentence is summary (1 if oracle sentence)
        :return segments_ids: array that indicates alternating segments with 0, 1
        :return cls_ids: array that indicates the position of CLS tokens
        :return src_txt: the source txt after pre-processing (array of strings)
        :return tgt_txt: the target text after pre-processing (string)
        """

        if len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if len(s) > self.args.min_src_ntokens]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        # if source is shorter than minimum
        if len(src) < self.args.min_nsents:
            return None
        # if labels has no elements
        if len(labels) == 0:
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS]'.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens + ['[SEP]'] + ['[CLS]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        # Segment IDs
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # CLS IDs
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_id]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
