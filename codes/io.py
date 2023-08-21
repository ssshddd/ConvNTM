import re
import numpy as np
import torch
import torch.utils.data
from collections import Counter


PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
SEP_WORD = '[SEP]'
PAD_ID = 0
SEP_ID = 102

class ConvNTMDataset(torch.utils.data.Dataset):
    def __init__(self, opt, examples, bow_dictionary, drop_few=True):
        
        keys = ['conv_str', 'conv_ids', 'conv_uttr_bow', 'conv_bow', 'uttr_num']
        
        if drop_few:
            filtered_examples = []

            for e in examples:
                filtered_example = {}
                if len(e['conv_bow']) < 3:
                    continue
                for k in keys:
                    filtered_example[k] = e[k]
                # if 'oov_list' in filtered_example:
                    # filtered_example['oov_number'] = len(filtered_example['oov_list'])

                filtered_examples.append(filtered_example)
        else:
            filtered_examples=examples

        self.examples = filtered_examples
        self.bow_dictionary = bow_dictionary
        self.opt = opt

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad_bow(self, input_list):
        bow_vocab = len(self.bow_dictionary)
        res_uttr_bow = np.zeros((len(input_list), self.opt.max_uttr_num, bow_vocab))
        for idx, bow_uttr in enumerate(input_list):
            for n in range(len(bow_uttr)):
                bow_k = [k for k, v in bow_uttr[n]]
                bow_v = [v for k, v in bow_uttr[n]]
                res_uttr_bow[idx, n, bow_k] = bow_v

        return torch.FloatTensor(res_uttr_bow)

    def collate_bow(self, batches):
        batch_bow = [b['conv_uttr_bow'] for b in batches]
        lens = [b['uttr_num'] for b in batches]
        return self._pad_bow(batch_bow), lens

    def collate_conv(self, batches):
        conv = [[b['conv_ids'],b['uttr_num']] for b in batches]
        return conv



def build_dataset_conv(opt, tokenized_convs, bow_dictionary, cur_bow):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    return_examples = []
    few_bow = 0

    for idx, (conv, bow) in enumerate(zip(tokenized_convs, cur_bow)):
        example = {}

        if conv[0].count(SEP_ID) <= opt.max_uttr_num:
            uttr_num = conv[0].count(SEP_ID)
            example['conv_str'] = conv[1]
            example['conv_ids'] = conv[0]
        else:
            uttr_num = opt.max_uttr_num
            conv_trunc, n_SEP, n_SEP_ID = [[], []], 0, 0
            for tok in conv[1]:
                if n_SEP < uttr_num:
                    conv_trunc[1].append(tok)
                    if tok == SEP_WORD:
                        n_SEP += 1

            
            for i in conv[0]:
                if n_SEP_ID < uttr_num:
                    conv_trunc[0].append(i)
                    if i == SEP_ID:
                        n_SEP_ID += 1

            example['conv_str'] = conv_trunc[1]
            example['conv_ids'] = conv_trunc[0]

        example['uttr_num'] = uttr_num
        assert bow.count('/sep/') >= uttr_num

        n_sep = 0
        example['conv_uttr_bow'] = [[]] * uttr_num
        bow_truncs, bow_uttr = [], []
        for w in bow:
            if n_sep < uttr_num:
                bow_uttr.append(w)
                if w == '/sep/':
                    example['conv_uttr_bow'][n_sep] = bow_dictionary.doc2bow(bow_uttr)
                    bow_truncs.extend(bow_uttr)
                    bow_uttr = []
                    n_sep += 1
        example['conv_bow'] = Counter(dict())
        for bow in example['conv_uttr_bow']:
            example['conv_bow'] += Counter(dict(bow))
        example['conv_bow'] = dict(example['conv_bow'])
        if len(example['conv_bow'].keys()) < 3:
            few_bow += 1
        
        return_examples.append(example)


    print("Find %d few bow" % few_bow)

    return return_examples



