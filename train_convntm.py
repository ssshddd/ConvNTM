import torch
import torch.nn as nn
import argparse
import logging
import json
import time
import numpy as np
import random
from itertools import chain
from torch.optim import Adam

import config
import train_model
from codes.model import ConvNTM, ConvSeqEncoder

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    opt.data = "processed_data/{}/".format(opt.data_tag)
    opt.vocab = opt.data
    opt.exp = 'trial.' + opt.data_tag if opt.trial else opt.data_tag

    print(opt.data_tag)
    if 'dailydialogues' in opt.data_tag:
        opt.max_uttr_len = 150
        opt.max_uttr_num = 25
        opt.speaker_num = 2
    elif 'emp' in opt.data_tag:
        opt.max_uttr_len = 150
        opt.max_uttr_num = 8
        opt.speaker_num = 2
    else:
        print('Wrong data_tag!!')
        return

    size_tag = "h_size{}".format(opt.trm_hidden_size)
    # only train ntm
    if opt.only_train_ntm:
        assert opt.ntm_warm_up_epochs > 0 and not opt.load_pretrain_ntm
        opt.use_speaker_reps = False
        opt.joint_train = False
        opt.exp += '.topic_num{}'.format(opt.topic_num)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print("Only training the ntm for %d epochs and save it to %s" % (opt.ntm_warm_up_epochs, opt.model_path))
        return opt

    # joint train settings
    if opt.joint_train:
        opt.exp += '.joint_train'
        if opt.joint_train_strategy != 'p_1_joint':
            opt.exp += '.' + opt.joint_train_strategy
            opt.pre_enc = int(opt.joint_train_strategy.split('_')[1])
            opt.exp += '.pre_enc{}'.format(opt.pre_enc)
            if opt.joint_train_strategy.split('_')[-1] != 'joint':
                opt.iterate_train_ntm = True

    opt.exp += '.topic_num{}'.format(opt.topic_num)

    if opt.topic_type == 'z':
        opt.exp += '.z_topic'

    if opt.load_pretrain_ntm:
        opt.check_pt_ntm_model_path = '[MODEL_NTM]'
        opt.check_pt_model_path = '[MODEL_ENCODER]'
        has_topic_num = [t for t in opt.check_pt_ntm_model_path.split('.') if 'topic_num' in t]
        if len(has_topic_num) != 0:
            assert opt.topic_num == int(has_topic_num[0].replace('topic_num', ''))


    # fill time into the name
    if opt.model_path.find('%s') > 0:
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('Model_PATH : ' + opt.model_path)

    return opt


def init_optimizers(model, ntm_model, opt):
    optimizer_encoder = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_encoder, optimizer_ntm, optimizer_whole


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
        word2idx, idx2word, vocab_size, bow_dictionary = load_data_and_vocab(opt, load_train=True)
        opt.bow_vocab_size = len(bow_dictionary)
        opt.vocab_size = vocab_size
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        start_time = time.time()
        model = ConvSeqEncoder(opt).to(opt.device)
        ntm_model = ConvNTM(opt, hidden_dim=opt.ntm_dim).to(opt.device)



        if torch.cuda.device_count() > 1:
            logging.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            model = nn.DataParallel(model, dim = 0)
            ntm_model = nn.DataParallel(ntm_model, dim = 0)
        elif torch.cuda.device_count() == 1:
            logging.info("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            logging.info("Let's use CPU/s!")

        logging.info(str(model), str(ntm_model))
        logging.info('model paramteres: ' + str(sum(param.numel() for param in model.parameters() if param.requires_grad==True)))
        logging.info('ntm model paramteres: ' + str(sum(param.numel() for param in ntm_model.parameters() if param.requires_grad==True)))
        logging.info('model paramteres: ' + str(sum(param.numel() for name, param in model.named_parameters() if param.requires_grad==True and '.emb.' not in name)))
        logging.info('model paramteres: ' + str(sum(param.numel() for name, param in model.named_parameters() if param.requires_grad==True and '.emb.' in name)))
        optimizer_encoder, optimizer_ntm, optimizer_whole = init_optimizers(model, ntm_model, opt)

        train_model.train_model(model,
                                ntm_model,
                                optimizer_encoder,
                                optimizer_ntm,
                                optimizer_whole,
                                train_data_loader,
                                valid_data_loader,
                                bow_dictionary,
                                train_bow_loader,
                                valid_bow_loader,
                                opt)

        training_time = time_since(start_time)

        logging.info('Time for training: %.1f' % training_time)

    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)


    opt.device = device

    if not opt.graph_flag:
        opt.exp = 'nograph_' + opt.exp
    if not opt.attn_flag:
        opt.exp = 'noattn_' + opt.exp
    if not opt.rnn_flag:
        opt.exp = 'nornn_' + opt.exp
    if opt.speaker_num == 1:
        opt.exp = 'nospeaker_' + opt.exp

    logging = config.init_logging(log_file=opt.model_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    print(opt)

    main(opt)
