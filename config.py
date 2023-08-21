import logging
import os
import sys
import time


def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def model_opts(parser):
    parser.add_argument('-seed', type=int, default=9527,
                        help="Random seed for the experiments")


    ## ConvSeqEncoder options
    # Transformer_Encoder options
    parser.add_argument('-trm_hidden_size', type=int, default=64,
                        help='Size of encoder hidden states')    
    parser.add_argument('-trm_dropout', type=float, default=0.1,
                        help="Dropout probability in transformer encoder")
    parser.add_argument('-n_trm_layer', type=int, default=2,
                        help='Number of transformer encoder layers')
    parser.add_argument('-n_head', type=int, default=2,
                        help='Number of attention heads')

    # RNN_Layer options
    parser.add_argument('-rnn_flag', default=True, action='store_true',
                        help="Use RNN_Layer or not")
    parser.add_argument('-rnn_hidden_size', type=int, default=64,
                        help='Size of encoder hidden states')
    parser.add_argument('-rnn_dropout', type=float, default=0,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-rnn_net', type=str, default='lstm',
                        choices=['lstm', 'gru', 'rnn'],
                        help="The gate type to use in the RNNs")
    parser.add_argument('-bidirectional', default=True,
                        action="store_true",
                        help="whether the encoder is bidirectional")

    # Attention_Layer options
    parser.add_argument('-attn_flag', default=True, action='store_true',
                        help="Use Attention_Layer or not")
    parser.add_argument('-attn_hidden_size', type=int, default=64,
                        help='Size of encoder hidden states')
    # Graph Encoder options
    parser.add_argument('-graph_flag', default=True, action='store_true',
                        help="Use Graph Encoder or not")
    parser.add_argument('-gnn_hidden_size', type=int, default=64,
                        help='Size of gnn hidden states')
    parser.add_argument('-gnn_net', type=str, default='gcn',
                        choices=['gcn', 'graphconv'],
                        help="The gate type to use in the GNNs")
    parser.add_argument('-bn_gnn', default=True, action='store_true',
                        help="Use BN in Graph Encoder or not")

    ## ConvNTM options
    parser.add_argument('-ntm_dim', type=int, default=500)
    parser.add_argument('-use_speaker_reps', default=True, action='store_true',
                        help="Use speaker representations in the topic model")
    parser.add_argument('-speaker_num', type=int, default=2)
    parser.add_argument('-topic_num', type=int, default=20)
    parser.add_argument('-topic_type', default='g', choices=['z', 'g'], help='use latent variable z or g as topic')

def preprocess_opts(parser):
    parser.add_argument('-data_dir', help='The source file of the data')

    # Dictionary options
    parser.add_argument('-vocab_size', type=int, default=20000,
                        help="Size of the source vocabulary")
    parser.add_argument('-bow_vocab', type=int, default=10000,
                        help="Size of the bow dictionary")
    parser.add_argument('-no_below', type=int, default=3,
                        help="no_below of the bow dictionary")
    parser.add_argument('-no_above', type=int, default=0.2,
                        help="no_above of the bow dictionary")
    
    # Utterances of Conversations
    parser.add_argument('-max_uttr_num', type=int, default=35,
                        help="Max number of the utterances per conversation")
    # parser.add_argument('-max_conv_len', type=int, default=2000,
    #                     help="Max length of the conversation sequence")
    parser.add_argument('-max_uttr_len', type=int, default=150,
                        help="Max length of one utterance sequence")


def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data_tag', type=str, required=True)
    parser.add_argument('-data',  # required=True,
                        help="Path prefix to train_convs.pt from preprocess.py")
    parser.add_argument('-vocab',  # required=True,
                        help="Path prefix to vocab.pt from preprocess.py")
    parser.add_argument('-load_pretrain_ntm', default=False, action='store_true')
    parser.add_argument('-only_eval', default=False, action='store_true')

    parser.add_argument('-custom_data_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-custom_vocab_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-vocab_filename_suffix', default='',
                        help='')
    parser.add_argument('-data_filename_suffix', default='',
                        help='')


    # Init options
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of training epochs')
    # parser.add_argument('-start_epoch', type=int, default=1,
                        # help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # # Pretrained word vectors
    # parser.add_argument('-pre_word_vecs_enc',
    #                     help="""If a valid path is specified, then this will load
    #                     pretrained word embeddings on the encoder side.
    #                     See README for specific formatting instructions.""")


    # Optimization options
    parser.add_argument('-batch_size', type=int, default=100,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    parser.add_argument('-max_grad_norm', type=float, default=1,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-fast_lr', type=float, default=0.1,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")

    parser.add_argument('-warmup', default=False, action='store_true')

    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")

    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    # parser.add_argument('-decay_method', type=str, default="",
    #                     choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-early_stop_tolerance', type=int, default=50,
                        help="Stop training if it doesn't improve any more for several rounds of validation")

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")

    parser.add_argument('-report_every', type=int, default=10,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    # parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
    #                     help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="model/%s.%s",
                        help="Path of checkpoints.")


    parser.add_argument('-trial', default=False, action='store_true')
    parser.add_argument('-joint_train', default=True, action='store_true')
    parser.add_argument('-joint_train_strategy', default='p_0_joint')
    parser.add_argument('-iterate_train_ntm', default=False, action='store_true')
    parser.add_argument('-pre_enc', type=int, default=1,
                         help='number of epochs for training encoder before joint train')
    parser.add_argument('-save_each_epoch', default=False, action='store_true')
    parser.add_argument('-only_train_ntm', default=False, action='store_true')
    parser.add_argument('-check_pt_ntm_model_path', type=str)
    parser.add_argument('-check_pt_model_path', type=str)
    parser.add_argument('-ntm_warm_up_epochs', type=int, default=200)
    parser.add_argument('-target_sparsity', type=float, default=0.85,
                        help="Target sparsity for ntm model")
    parser.add_argument('-w_KL', type=float, default=0.01,
                        help="weight for KL loss")
    parser.add_argument('-w_recbow', type=float, default=0.001,
                        help="weight for KL loss")
    parser.add_argument('-add_co_matirx', default=True, action='store_true')
    parser.add_argument('-w_rec_M', type=float, default=10,
                        help="weight for rec_M loss")
    parser.add_argument('-target_rec_M', type=float, default=32,
                        help="Target rec_M for ntm model 32/31.385(dd)  22.76/22.132(emp)")

    parser.add_argument('-use_pretrained', default=False, action='store_true')
    parser.add_argument('-add_pretrained', default=False, action='store_true')
        

'''
target_rec_M Recommend Settings
Dailydialogues
20: 32.007378
10: 22.761194/22.746965
30: 41.560604/41.561565
50: 60.967541
70: 80.577759/80.576942
100: 110.378664/110.193764
EMP
20: 31.374624
10: 22.140675
30: 40.949936/40.949768
50: 60.462128
70: 80.087471/80.090469
100: 109.731911/109.718285
'''