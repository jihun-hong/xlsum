import argparse

from others.utils import init_logger, str2bool
from prepro import data_converter


def process(args):
    assert args.max_nsents > args.min_nsents
    assert args.max_src_ntokens > args.min_src_ntokens

    mode = args.mode.lower()
    assert mode in ['tokenize', 'json', 'xlnet', 'all']

    if mode == 'tokenize':
        data_converter.tokenize(args)
    elif mode == 'json':
        data_converter.format_json(args)
    elif mode == 'xlnet':
        data_converter.format_xlnet(args)
    elif mode == 'all':
        data_converter.tokenize(args)
        data_converter.format_json(args)
        data_converter.format_xlnet(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", default='all', type=str, help='tokenize, json, xlnet, or all')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or'
                                                                         'combination, combination will generate more'
                                                                         'accurate oracles but take much longer time.')

    # Path configuration.
    parser.add_argument("-map_path", default='../urls/')
    parser.add_argument("-raw_path", default='../raw_data/raw_files/')
    parser.add_argument("-tokenized_path", default='../raw_data/tokenized_files/')
    parser.add_argument("-json_path", default='../raw_data/json_files/cnndm')
    parser.add_argument("-save_path", default='../xlnet_data/')

    # Batch configuration.
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=1500, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("-log_file", default='../logs/preprocess.log')
    parser.add_argument('-dataset', default='', help='train, valid or test, default will process all datasets')
    parser.add_argument('-n_cpus', default=2, type=int)

    # Stanford CoreNLP.
    parser.add_argument("-tokenizer_dir", type=str, default="../stanford-corenlp/")
    parser.add_argument("-tokenizer_date", type=str, default="2018-10-05")
    parser.add_argument("-tokenizer_ver", type=str, default="3.9.2")

    args = parser.parse_args()
    init_logger(args.log_file)

    process(args)
