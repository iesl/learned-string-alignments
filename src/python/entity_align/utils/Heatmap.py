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

import argparse

import matplotlib
import numpy as np
import torch
from torch.autograd import Variable

from entity_align.model.Vocab import Vocab
from entity_align.utils.Config import Config

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs

def load_data(input_file, vocab, config):
    with codecs.open(input_file, "r", "UTF-8", errors="ignore") as inp:
        sources = []
        sources_lengths = []
        targets = []
        targets_lengths = []
        ct = -1
        print("begin loading data")
        for line in inp:
            print(line)
            ct += 1
            line = line
            split = line.split("\t") #source, pos, negative
            sources.append(np.asarray(vocab.to_ints(split[0])))
            sources_lengths.append([min(config.max_string_len,len(split[0]))])
            targets.append(np.asarray(vocab.to_ints(split[1])))
            targets_lengths.append([min(config.max_string_len,len(split[1]))])
        print("complete loading data")
        sources = np.asarray(sources)
        targets = np.asarray(targets)
        sources_lengths = np.asarray(sources_lengths)
        targets_lengths = np.asarray(targets_lengths)
        return sources, targets, sources_lengths, targets_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_file", required=True)
    parser.add_argument("-m", "--model_path", default=True)
    parser.add_argument("-v", "--vocab", default=True)
    parser.add_argument("-c", "--config", default=True)

    args = parser.parse_args()
    tfp = args.test_file
    config = Config(args.config)
    vocab = Vocab(args.vocab, config.max_string_len)
    sources, targets, sources_lengths, targets_lengths = load_data(tfp, vocab, config)
    if config.bidirectional == True:
        num_directions = 2
    else:
        num_directions = 1
    model = torch.load(args.model_path)
    model.h0_dev = Variable(torch.zeros(num_directions, len(sources), config.rnn_hidden_size).cuda(), requires_grad=False)
    model.c0_dev = Variable(torch.zeros(num_directions, len(sources), config.rnn_hidden_size).cuda(), requires_grad=False)
    scores = model.print_mm(sources, targets, sources_lengths, targets_lengths).cpu().data.numpy()
    max_scores = np.max(scores)
    min_scores = np.min(scores)
    for idx in range(0, len(scores)):
        scores[idx][sources_lengths[idx]] = max_scores
        scores[idx][sources_lengths[idx] + 1] = min_scores
    for idx in range(0, len(scores)):
        fig = plt.figure()
        max_interesting = int(max(sources_lengths[idx], targets_lengths[idx])) + 2
        my_yticks = list(vocab.to_string(sources[idx]))
        plt.yticks(range(sources_lengths[idx]), my_yticks)
        my_xticks = list(vocab.to_string(targets[idx]))
        plt.xticks(range(targets_lengths[idx]), my_xticks)
        plt.imshow(scores[idx][:max_interesting+1, :max_interesting+1], cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        fig.savefig("images/im{}.png".format(idx))
