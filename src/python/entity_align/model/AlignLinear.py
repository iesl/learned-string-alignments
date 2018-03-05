"""
Copyright (C) 2017-2018 University of Massachusetts Amherst.
This file is part of "learned-string-alignments"
http://github.com/iesl/learned-string-alignments
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss


#This model corresponds to AlignWeights in our paper
#First, strings converted to list of character embeddings
#Then, lstm runs over character embeddings
#lstm embeddings at last time stamp matrix multiplied
#Finally, matrix dot multiplied by block of weights and summed and scored
class AlignLinear(torch.nn.Module):
    def __init__(self,config,vocab):
        super(AlignLinear, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.size+1, config.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(config.embedding_dim, config.rnn_hidden_size, 1, bidirectional = config.bidirectional, batch_first = True)
        self.loss = BCEWithLogitsLoss()
        if self.config.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # Variables for initial states of LSTM (these are different for train and dev because dev might be of different batch sizes)
        self.h0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.h0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.align_weights = nn.Parameter(torch.randn(config.max_string_len, config.max_string_len).cuda(),requires_grad=True)
        self.ones = Variable(torch.ones(config.batch_size, 1).cuda())


    def compute_loss(self,source,pos,neg, source_len,pos_len,neg_len):
        """ Compute the loss (BPR) for a batch of examples
        :param source: Entity mentions
        :param pos: True aliases of the Mentions
        :param neg: False aliases of the Mentions
        :param source_len: lengths of mentions
        :param pos_len: lengths of positives
        :param neg_len: lengths of negatives
        :return:
        """
        source_embed, src_mask = self.embed(source,source_len)
        pos_embed, pos_mask = self.embed(pos,pos_len)
        neg_embed, neg_mask = self.embed(neg,neg_len)
        loss = self.loss(
            self.score_pair_train(source_embed , pos_embed, src_mask, pos_mask)
            - self.score_pair_train(source_embed , neg_embed, src_mask, neg_mask),
            self.ones)

        return loss

    def print_mm(self, src, tgt, src_len, tgt_len):
        """ Prints the matrix multiplication of the two embeddings.
        This function is useful for creating a heatmap for figures.
        :param src: Entity mentions
        :param tgt: Entity mentions
        :param src_len: lengths of src mentions
        :param neg_len: lengths of tgt mentions
        :return:
        """
        source_embed = self.embed_dev(src, src_len)
        target_embed = self.embed_dev(tgt, tgt_len)
        return torch.bmm(source_embed,torch.transpose(target_embed, 2, 1))

    def score_pair_train(self,src,tgt, src_mask, tgt_mask):
        """
        :param src: Batchsize by Max_String_Length
        :param tgt: Batchsize by Max_String_Length
        :param src_mask: Batchsize by Max_String_Length, binary mask corresponding to length of underlying str
        :param tgt_mask: Batchsize by Max_String_Length, binary mask corresponding to length of underlying str
        :return: Batchsize by 1
        """
        multpld = torch.bmm(src,torch.transpose(tgt, 2, 1))
        src_mask = src_mask.unsqueeze(dim=2)
        tgt_mask = tgt_mask.unsqueeze(dim=1)
        mat_mask = torch.bmm(src_mask, tgt_mask)
        multpld = torch.mul(multpld, mat_mask)
        output = torch.sum(self.align_weights.expand_as(multpld) * multpld, dim=1,keepdim=True)
        output = torch.sum(output, dim=2,keepdim=True)
        output = torch.squeeze(output, dim=2)
        return output

    def embed(self,string_mat, string_len):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
        string_mat = torch.from_numpy(string_mat).cuda()
        mask = Variable(torch.cuda.ByteTensor((string_mat > 0)).float())
        embed_token = self.embedding(Variable(string_mat))
        final_emb, final_hn_cn = self.rnn(embed_token, (self.h0, self.c0))
        return final_emb, mask

    def embed_dev(self, string_mat, string_len, print_embed=False, batch_size = None):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
        string_mat = torch.from_numpy(string_mat).cuda()
        mask = Variable(torch.cuda.ByteTensor((string_mat > 0)).float())
        if not batch_size:
            this_batch_size = self.config.dev_batch_size
            this_h0 = self.h0_dev
            this_c0 = self.c0_dev
        else:
            print("irregular batch size {}".format(batch_size))
            this_batch_size = batch_size
            this_h0 = Variable(torch.zeros(self.num_directions, batch_size,
                                           self.config.rnn_hidden_size).cuda(),
                               requires_grad=False)
            this_c0 = Variable(torch.zeros(self.num_directions, batch_size,
                                           self.config.rnn_hidden_size).cuda(),
                               requires_grad=False)
        embed_token = self.embedding(Variable(string_mat))
        if print_embed==True:
            return embed_token
        final_emb, final_hn_cn = self.rnn(embed_token, (this_h0, this_c0))
        return final_emb, mask

    def score_dev_test_batch(self,batch_queries,
                             batch_query_lengths,
                             batch_targets,
                             batch_target_lengths,
                             batch_size):
        if batch_size == self.config.dev_batch_size:
            source_embed,source_mask = self.embed_dev(batch_queries, batch_query_lengths)
            target_embed,target_mask = self.embed_dev(batch_targets, batch_target_lengths)
        else:
            source_embed,source_mask = self.embed_dev(batch_queries, batch_query_lengths,batch_size=batch_size)
            target_embed,target_mask = self.embed_dev(batch_targets, batch_target_lengths,batch_size=batch_size)
        scores = self.score_pair_train(source_embed, target_embed,source_mask,target_mask)
        return scores
