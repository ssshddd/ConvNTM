import torch
import logging
from codes.io import *
from torch.utils.data import DataLoader


def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from : %s" % (opt.vocab))
    if not opt.custom_vocab_filename_suffix:
        word2idx, idx2word, vocab, bow_dictionary = torch.load(opt.vocab + '/vocab.pt', 'rb')
    else:
        word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.%s.pt' % opt.vocab_filename_suffix, 'rb')
    # assign vocab to opt
    # opt.word2idx = word2idx
    # opt.idx2word = idx2word
    # opt.vocab = vocab
    # opt.bow_dictionary = bow_dictionary
    logging.info('#(vocab size)=%d' % vocab)
    # logging.info('#(vocab used)=%d' % opt.vocab_size)
    logging.info('#(bow dictionary size)=%d' % len(bow_dictionary))

    return word2idx, idx2word, vocab, bow_dictionary


def load_data_and_vocab(opt, load_train=True):
    # load vocab
    word2idx, idx2word, vocab, bow_dictionary = load_vocab(opt)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)

    if load_train:  # load training dataset
        if not opt.custom_data_filename_suffix:
            train_convs = torch.load(opt.data + '/train_convs.pt', 'rb')
        else:
            train_convs = torch.load(opt.data + '/train_convs.%s.pt' % opt.data_filename_suffix, 'rb')
        train_conv_dataset = ConvNTMDataset(opt, train_convs, bow_dictionary)
        train_loader = DataLoader(dataset=train_conv_dataset,
                                  collate_fn=train_conv_dataset.collate_conv,
                                  num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                  pin_memory=True, shuffle=False)
        train_bow_loader = DataLoader(dataset=train_conv_dataset,
                                      collate_fn=train_conv_dataset.collate_bow,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                      pin_memory=True, shuffle=False)

        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

        if not opt.custom_data_filename_suffix:
            valid_convs = torch.load(opt.data + '/valid_convs.pt', 'rb')
        else:
            valid_convs = torch.load(opt.data + '/valid_convs.%s.pt' % opt.data_filename_suffix, 'rb')
        valid_conv_dataset = ConvNTMDataset(opt, valid_convs, bow_dictionary)
        valid_loader = DataLoader(dataset=valid_conv_dataset,
                                  collate_fn=valid_conv_dataset.collate_conv,
                                  num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                  pin_memory=True, shuffle=False)
        valid_bow_loader = DataLoader(dataset=valid_conv_dataset,
                                      collate_fn=valid_conv_dataset.collate_bow,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                      pin_memory=True, shuffle=False)

        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))

        return train_loader, train_bow_loader, valid_loader, valid_bow_loader, word2idx, idx2word, vocab, bow_dictionary
    else:
        if not opt.custom_data_filename_suffix:
            test_convs = torch.load(opt.data + '/test_convs.pt', 'rb')
        else:
            test_convs = torch.load(opt.data + '/test_convs.%s.pt' % opt.data_filename_suffix, 'rb')
        test_conv_dataset = ConvNTMDataset(opt, test_convs, bow_dictionary)
        test_loader = DataLoader(dataset=test_conv_dataset,
                                 collate_fn=test_conv_dataset.collate_conv,
                                 num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                 pin_memory=True, shuffle=False)
        test_bow_loader = DataLoader(dataset=test_conv_dataset,
                                     collate_fn=test_conv_dataset.collate_bow,
                                     num_workers=opt.batch_workers, batch_size=opt.batch_size, 
                                     pin_memory=True, shuffle=False)

        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

        return test_loader, test_bow_loader, word2idx, idx2word, vocab, bow_dictionary



def load_data_and_vocab_all(opt):
    # load vocab
    word2idx, idx2word, vocab, bow_dictionary = load_vocab(opt)


    if not opt.custom_data_filename_suffix:
        train_convs = torch.load(opt.data + '/train_convs.pt', 'rb')
    else:
        train_convs = torch.load(opt.data + '/train_convs.%s.pt' % opt.data_filename_suffix, 'rb')
    train_conv_dataset = ConvNTMDataset(opt, train_convs, bow_dictionary, drop_few=False)
    train_loader = DataLoader(dataset=train_conv_dataset,
                                collate_fn=train_conv_dataset.collate_conv,
                                num_workers=opt.batch_workers, batch_size=1, 
                                pin_memory=True, shuffle=False)
    train_bow_loader = DataLoader(dataset=train_conv_dataset,
                                    collate_fn=train_conv_dataset.collate_bow,
                                    num_workers=opt.batch_workers, batch_size=1, 
                                    pin_memory=True, shuffle=False)

    logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

    if not opt.custom_data_filename_suffix:
        valid_convs = torch.load(opt.data + '/valid_convs.pt', 'rb')
    else:
        valid_convs = torch.load(opt.data + '/valid_convs.%s.pt' % opt.data_filename_suffix, 'rb')
    valid_conv_dataset = ConvNTMDataset(opt, valid_convs, bow_dictionary, drop_few=False)
    valid_loader = DataLoader(dataset=valid_conv_dataset,
                                collate_fn=valid_conv_dataset.collate_conv,
                                num_workers=opt.batch_workers, batch_size=1, 
                                pin_memory=True, shuffle=False)
    valid_bow_loader = DataLoader(dataset=valid_conv_dataset,
                                    collate_fn=valid_conv_dataset.collate_bow,
                                    num_workers=opt.batch_workers, batch_size=1, 
                                    pin_memory=True, shuffle=False)

    logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))


    if not opt.custom_data_filename_suffix:
        test_convs = torch.load(opt.data + '/test_convs.pt', 'rb')
    else:
        test_convs = torch.load(opt.data + '/test_convs.%s.pt' % opt.data_filename_suffix, 'rb')
    test_conv_dataset = ConvNTMDataset(opt, test_convs, bow_dictionary, drop_few=False)
    test_loader = DataLoader(dataset=test_conv_dataset,
                                collate_fn=test_conv_dataset.collate_conv,
                                num_workers=opt.batch_workers, batch_size=1, 
                                pin_memory=True, shuffle=False)
    test_bow_loader = DataLoader(dataset=test_conv_dataset,
                                    collate_fn=test_conv_dataset.collate_bow,
                                    num_workers=opt.batch_workers, batch_size=1, 
                                    pin_memory=True, shuffle=False)

    logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

    return train_loader, train_bow_loader, valid_loader, valid_bow_loader, test_loader, test_bow_loader, word2idx, idx2word, vocab, bow_dictionary