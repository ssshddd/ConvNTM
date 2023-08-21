import torch
import torch.nn as nn
import torch.nn.functional as F



class Embeddings(nn.Module):
    """Construct token, position embeddings.
    """
    def __init__(self, hidden_size, vocab_size, max_position_size, padding_idx, dropout_rate, device, use_pretrained=False):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_size+1, hidden_size, padding_idx=padding_idx)
        # self.segment_embeddings = nn.Embedding(max_segment, hidden_size, padding_idx=word_emb_padding_idx)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_seq_len = max_position_size
        self.padding_idx = padding_idx
        self.device = device

        self.use_pretrained = use_pretrained
        if use_pretrained:
            from transformers import BertModel
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_list, cls_idx=101, sep_idx=102, use_pretrained=False):
        uttr_nums = [un for idx, un in input_list]
        un = 0
        uttr_ids_list = torch.ones(sum(uttr_nums), self.max_seq_len, dtype=torch.long, device=self.device) * self.padding_idx
        position_ids_list = torch.ones_like(uttr_ids_list, device=self.device) * self.padding_idx
        for i, (conv_ids, uttr_num) in enumerate(input_list):
            assert uttr_num == conv_ids.count(sep_idx)
            uttr_tmp = []
            for idx in conv_ids:
                if un < sum(uttr_nums):
                    uttr_tmp.append(idx)
                    if idx == sep_idx:
                        if len(uttr_tmp) > self.max_seq_len:
                            uttr_tmp = uttr_tmp[:self.max_seq_len-1] + [sep_idx]
                        uttr_ids_list[un, :len(uttr_tmp)] = torch.LongTensor(uttr_tmp)
                        position_ids_list[un, :len(uttr_tmp)] = torch.arange(1, len(uttr_tmp)+1)
                        un += 1
                        uttr_tmp = [cls_idx]
        padding_mask = (uttr_ids_list == self.padding_idx)

        words_embeddings = self.word_embeddings(uttr_ids_list)
        position_embeddings = self.position_embeddings(position_ids_list)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.use_pretrained:
            with torch.no_grad():
                bert_uttr_reps = self.bert_model(input_ids=uttr_ids_list).last_hidden_state[:,0,:]
            return embeddings, padding_mask, bert_uttr_reps
        return embeddings, padding_mask


class Transformer_Encoder(nn.Module):
    def __init__(self, n_layer, hidden_size, num_attention_heads, vocab_size, max_position_size, device, padding_idx=0, dropout=0.1, use_pretrained=False):
        super(Transformer_Encoder, self).__init__()
        self.emb = Embeddings(hidden_size, vocab_size, max_position_size, padding_idx, dropout, device, use_pretrained)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer, encoder_norm)
        # self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.use_pretrained = use_pretrained

    def forward(self, input_list):
        if self.use_pretrained:
            emb, padding_mask, bert_uttr_reps = self.emb(input_list) 
        else:
            emb, padding_mask = self.emb(input_list)
        # uttr_last_num = torch.LongTensor([un for idx, un in input_list]).cumsum(0).tolist()
        emb = self.encoder(emb.transpose(0,1), src_key_padding_mask=padding_mask).transpose(0,1) # [sum_uttr_num, max_uttr_len, hidden_size]

        if self.use_pretrained:
            return emb[:,0,:], bert_uttr_reps
        return emb[:,0,:]   # [sum_uttr_num, hidden_size]


class RNN_Layer(nn.Module):
    def __init__(self, hidden_dim, is_bi_rnn, net='lstm', dropout=0.0):
        super(RNN_Layer,self).__init__()
        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        if net == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=is_bi_rnn, dropout=dropout)
        if net == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=is_bi_rnn, dropout=dropout)
        if net == 'rnn':
            self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True, bidirectional=is_bi_rnn, dropout=dropout)
        self.net = net

    def forward(self, inputs, lens):
        inputs_packed = nn.utils.rnn.pack_padded_sequence(inputs, lens, batch_first=True, enforce_sorted=False)
        if self.net == 'lstm':
            outputs_packed, (hn, cn) = self.rnn(inputs_packed)
        else:
            outputs_packed, hn = self.rnn(inputs_packed)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)[0]  # [bs, max_uttr_num, hidden_size]
        return outputs


class Attention_Layer(nn.Module):
    # self-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Attention_Layer,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn
        hidden_size = hidden_dim * 2 if is_bi_rnn else hidden_dim
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias = False)
     
    def forward(self, inputs, lens):
        size = inputs.size()  # [bs, max_len, hidden_size]
        Q = self.Q_linear(inputs) 
        K = self.K_linear(inputs).permute(0, 2, 1)
        V = self.V_linear(inputs)
        
        max_len = max(lens)
        sentence_lengths = torch.Tensor(lens)
        mask = torch.arange(max_len)[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(1).expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]
        
        padding_num = -torch.ones_like(mask).float() * 1e9

        alpha = torch.matmul(Q, K)
        alpha = torch.where(mask.to(alpha.device), alpha, padding_num.to(alpha.device))
        alpha = F.softmax(alpha, dim=-1)  # [bs, max_uttr_num, max_uttr_num]
        out = torch.matmul(alpha, V)  # [bs, max_uttr_num, hidden_size]
        return out, alpha

from torch_geometric.nn import GatedGraphConv, GraphConv, GCNConv
class Graph_Encoder(nn.Module):
    def __init__(self, hidden_dim, is_bi_rnn, net='gcn', bn_gnn=True):
        super(Graph_Encoder,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn
        hidden_size = hidden_dim * 2 if is_bi_rnn else hidden_dim
        if net == 'gcn':
            self.gnn = GCNConv(hidden_size, hidden_size)
        if net == 'graphconv':
            self.gnn = GraphConv(hidden_size, hidden_size)
        if bn_gnn:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.bn_gnn = bn_gnn
             
    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        if self.bn_gnn:
            x = self.bn(x)
        x = torch.tanh(x)
        return x