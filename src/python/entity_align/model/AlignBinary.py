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

#This model corresponds to AlignBinary in our paper
#First, characters in strings converted to one-hot vectors
#Then, binary matrix created: 1 where chars are the same, 0 otherwise
#Finally, cnn detects features in that matrix and outputs similarity score
class AlignBinary(torch.nn.Module):
    def __init__(self,config,vocab):
        super(AlignBinary, self).__init__()
        self.config = config
        self.vocab = vocab
        # Embeddings hack to create one-hot vectors for everything in our vocab
        self.embedding = nn.Embedding(vocab.size+1, vocab.size+1, padding_idx=0)
        self.embedding.weight.data = torch.eye(vocab.size+1)
        self.embedding.weight.requires_grad = False
        #loss fn
        self.loss = BCEWithLogitsLoss()
        if self.config.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1


        # Define the CNN used to score the alignment matrix

        pool_output_height = int(np.floor(config.max_string_len/2.0)) #also output_width

        # Select # of layers / increasing or decreasing filter size based on config

        if config.num_layers == 3:
            self.num_layers = 3
            self.relu = nn.ReLU()
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
            convlyr = nn.Conv2d(1, config.filter_count, 3, padding=1, stride=1)
            self.align_weights = nn.Parameter(torch.randn(config.filter_count, pool_output_height, pool_output_height).cuda(),requires_grad=True)
        self.add_module("cnn",convlyr)

        # Define pooling
        self.pool = nn.MaxPool2d((2, 2), stride=2)
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
        convd = self.cnn(multpld.unsqueeze(1)) #need num channels
        if self.num_layers == 3:
            convd = self.relu(convd)
            convd = self.cnn2(convd)
            convd = self.relu(convd)
            convd = self.cnn3(convd)
        elif self.num_layers == 2:
            convd = self.relu(convd)
            convd = self.cnn2(convd)
        convd_after_pooling = self.pool(convd)
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
        return embed_token, mask

    def embed_dev(self, string_mat, string_len, print_embed=False, batch_size = None):
        """
        :param string_mat: Batch_size by max_string_len
        :return: batch_size by embedding dim
        """
        string_mat = torch.from_numpy(string_mat).cuda()
        mask = Variable(torch.cuda.ByteTensor((string_mat > 0)).float())
        embed_token = self.embedding(Variable(string_mat))
        if print_embed==True:
            return embed_token
        return embed_token, mask

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
