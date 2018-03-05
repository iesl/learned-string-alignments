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
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable


#This model corresponds to AlignCNN in our paper
#First, strings converted to list of character embeddings
#Then, lstm runs over character embeddings
#lstm embeddings at last time stamp matrix multiplied
#Finally, cnn detects features in that matrix and outputs similarity score
class AlignCNN(torch.nn.Module):
    def __init__(self,config,vocab):
        super(AlignCNN, self).__init__()
        self.config = config
        self.vocab = vocab

        # Character embeddings
        self.embedding = nn.Embedding(vocab.size+1, config.embedding_dim, padding_idx=0)

        # Sequence encoder of strings (LSTM)
        self.rnn = nn.LSTM(config.embedding_dim, config.rnn_hidden_size, 1, bidirectional = config.bidirectional, batch_first = True)

        if self.config.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # Variables for initial states of LSTM (these are different for train and dev because dev might be of different batch sizes)
        self.h0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0 = Variable(torch.zeros(self.num_directions, config.batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.h0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)
        self.c0_dev = Variable(torch.zeros(self.num_directions, config.dev_batch_size, config.rnn_hidden_size).cuda(), requires_grad=False)


        # Define the CNN used to score the alignment matrix
        pool_output_height = int(np.floor(config.max_string_len/2.0))

        # Select # of layers / increasing or decreasing filter size based on config
        if config.num_layers == 4:
            self.num_layers = 4
            self.relu = nn.ReLU()
            if config.increasing == True:
                convlyr = nn.Conv2d(1, config.filter_count, 3, padding=1, stride=1)
                convlyr2 = nn.Conv2d(config.filter_count, config.filter_count2, 5, padding=2, stride=1)
                convlyr3 = nn.Conv2d(config.filter_count2, config.filter_count3, 5, padding=2, stride=1)
                convlyr4 = nn.Conv2d(config.filter_count3, config.filter_count4, 7, padding=3, stride=1)
            else:
                convlyr = nn.Conv2d(1, config.filter_count, 7, padding=3, stride=1)
                convlyr2 = nn.Conv2d(config.filter_count, config.filter_count2, 5, padding=2, stride=1)
                convlyr3 = nn.Conv2d(config.filter_count2, config.filter_count3, 5, padding=2, stride=1)
                convlyr4 = nn.Conv2d(config.filter_count3, config.filter_count4, 3, padding=1, stride=1)
            self.add_module("cnn2",convlyr2)
            self.add_module("cnn3",convlyr3)
            self.add_module("cnn4",convlyr4)
            self.align_weights = nn.Parameter(torch.randn(config.filter_count3, pool_output_height, pool_output_height).cuda(),requires_grad=True)
        elif config.num_layers == 3:
            self.num_layers = 3
            self.relu = nn.ReLU()
            if config.increasing == True:
                convlyr = nn.Conv2d(1, config.filter_count, 5, padding=2, stride=1)
                convlyr2 = nn.Conv2d(config.filter_count, config.filter_count2, 5, padding=2, stride=1)
                convlyr3 = nn.Conv2d(config.filter_count2, config.filter_count3, 7, padding=3, stride=1)
            else:
                convlyr = nn.Conv2d(1, config.filter_count, 7, padding=3, stride=1)
                convlyr2 = nn.Conv2d(config.filter_count, config.filter_count2, 5, padding=2, stride=1)
                convlyr3 = nn.Conv2d(config.filter_count2, config.filter_count3, 5, padding=2, stride=1)
            self.add_module("cnn2",convlyr2)
            self.add_module("cnn3",convlyr3)
            self.align_weights = nn.Parameter(torch.randn(config.filter_count3, pool_output_height, pool_output_height).cuda(),requires_grad=True)
        elif config.num_layers == 2:
            self.num_layers = 2
            self.relu = nn.ReLU()
            convlyr = nn.Conv2d(1, config.filter_count, 5, padding=2, stride=1)
            convlyr2 = nn.Conv2d(config.filter_count, config.filter_count2, 3, padding=1, stride=1)
            self.add_module("cnn2",convlyr2)
            self.align_weights = nn.Parameter(torch.randn(config.filter_count2, pool_output_height, pool_output_height).cuda(),requires_grad=True)
        else:
            self.num_layers = 1
            convlyr = nn.Conv2d(1, config.filter_count, 7, padding=3, stride=1)
            self.align_weights = nn.Parameter(torch.randn(config.filter_count, pool_output_height, pool_output_height).cuda(),requires_grad=True)
        self.add_module("cnn",convlyr)
        # Define pooling
        self.pool = nn.MaxPool2d((2, 2), stride=2)

        # Vector of ones (used for loss)
        self.ones = Variable(torch.ones(config.batch_size, 1).cuda())

        # Loss
        self.loss = BCEWithLogitsLoss()

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
        source_embed, source_mask = self.embed_dev(src, src_len)
        target_embed, target_mask = self.embed_dev(tgt, tgt_len)
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
        convd = self.cnn(multpld.unsqueeze(1)) #need num channels
        if self.num_layers > 1:
            convd = self.relu(convd)
            convd = self.cnn2(convd)
        if self.num_layers > 2:
            convd = self.relu(convd)
            convd = self.cnn3(convd)
        if self.num_layers > 3:
            convd = self.relu(convd)
            convd = self.cnn4(convd)
        convd_after_pooling = self.pool(convd)
        #print(convd_after_pooling.size())
        #print(self.align_weights.size())
        output = torch.sum(self.align_weights.expand_as(convd_after_pooling) * convd_after_pooling, dim=3,keepdim=True)
        output = torch.sum(output, dim=2,keepdim=True)
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        output = torch.sum(output, dim=1,keepdim=True)

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
