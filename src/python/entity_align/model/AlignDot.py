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

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from entity_align.utils.Util import row_wise_dot

#This model corresponds to AlignDot in our paper
#First, strings converted to list of character embeddings
#Then, lstm runs over character embeddings
#Finally, lstm embeddings at last time stamp dot producted together and judgement made off dot product
class AlignDot(torch.nn.Module):
    def __init__(self,config,vocab):
        super(AlignDot, self).__init__()
        self.config = config
        self.vocab = vocab

        # Character embeddings
        self.embedding = nn.Embedding(vocab.size+1, config.embedding_dim, padding_idx=0)

        # Sequence encoder of strings (LSTM)
        self.rnn = nn.LSTM(config.embedding_dim, config.rnn_hidden_size, 1, bidirectional = config.bidirectional)

        self.loss = BCEWithLogitsLoss()
        #self.loss = nn.MarginRankingLoss(margin=1) for margin loss
        if self.config.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # Variables for initial states of LSTM (these are different for train and dev because dev might be of different batch sizes)
        self.h0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.h0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.ones = Variable(torch.ones(config.batch_size,1).cuda())

    def print_loss(self,source,pos,neg, source_len,pos_len,neg_len):
        source_embed = self.embed(source,source_len)
        pos_embed = self.embed(pos,pos_len)
        neg_embed = self.embed(neg,neg_len)
        print(row_wise_dot(source_embed , pos_embed)- row_wise_dot(source_embed , neg_embed ))


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
        source_embed = self.embed(source,source_len)
        pos_embed = self.embed(pos,pos_len)
        neg_embed = self.embed(neg,neg_len)
        loss = self.loss(
            row_wise_dot(source_embed , pos_embed )
            - row_wise_dot(source_embed , neg_embed ),
            self.ones)
        #For margin loss:
        #loss = self.loss(
        #    row_wise_dot(source_embed, pos_embed), row_wise_dot(source_embed, neg_embed),
        #    self.ones
        #)
        return loss

    def score_pair(self,source, target, source_len, target_len):
        """

        :param source: Batchsize by Max_String_Length
        :param target: Batchsize by Max_String_Length
        :return: Batchsize by 1
        """
        source_embed = self.embed_dev(source, source_len)
        target_embed = self.embed_dev(target, target_len)
        scores = row_wise_dot(source_embed, target_embed)
        return scores

    def embed(self,string_mat,string_len):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
        string_mat = torch.from_numpy(np.transpose(string_mat)).cuda()
        indices = Variable(string_mat) #torch.cuda.LongTensor(np.transpose(string_mat)))
        string_len = torch.from_numpy(string_len).cuda()
        lengths = Variable(string_len.squeeze() * self.config.batch_size)
        embed_token = self.embedding(indices)
        final_emb, final_hn = self.rnn(embed_token, (self.h0, self.c0))
        reshaped = final_emb.resize(self.config.batch_size * self.config.max_string_len, self.config.rnn_hidden_size * self.num_directions)
        offset = Variable(torch.cuda.LongTensor([x for x in range(0, self.config.batch_size)]))
        lookup = lengths + offset
        last_state = reshaped.index_select(0,lookup)
        return last_state

    def embed_dev(self,string_mat,string_len, print_embed=False,batch_size=None):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
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
        string_mat = torch.from_numpy(np.transpose(string_mat)).cuda()
        indices = Variable(string_mat)
        string_len = torch.from_numpy(string_len).cuda()
        lengths = Variable(string_len.squeeze() * this_batch_size)
        embed_token = self.embedding(indices)
        if print_embed==True:
            return embed_token
        final_emb, final_hn = self.rnn(embed_token, (this_h0,this_c0))
        if self.config.bidirectional == True:
            reshaped = final_emb.resize(this_batch_size * self.config.max_string_len, self.config.rnn_hidden_size * self.num_directions)
        else:
            reshaped = final_emb.resize(this_batch_size * self.config.max_string_len, self.config.rnn_hidden_size)
        offset = Variable(torch.cuda.LongTensor([x for x in range(0, this_batch_size)]))
        lookup = lengths + offset
        last_state = reshaped.index_select(0, lookup)
        return last_state

    def score_dev_test_batch(self,batch_queries,
                             batch_query_lengths,
                             batch_targets,
                             batch_target_lengths,
                             batch_size):
        if batch_size == self.config.dev_batch_size:
            source_embed = self.embed_dev(batch_queries, batch_query_lengths)
            target_embed = self.embed_dev(batch_targets, batch_target_lengths)
        else:
            source_embed = self.embed_dev(batch_queries, batch_query_lengths,batch_size=batch_size)
            target_embed = self.embed_dev(batch_targets, batch_target_lengths,batch_size=batch_size)
        scores = row_wise_dot(source_embed, target_embed)
        return scores
