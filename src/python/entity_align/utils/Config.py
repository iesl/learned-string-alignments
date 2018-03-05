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

import json
import random
import os

class Config(object):
    def __init__(self,filename=None):
        self.learning_rate = 0.0001
        self.l2penalty = 10.0
        self.vocab_file = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None
        self.num_minibatches = 100000
        self.batch_size = 100
        self.dev_batch_size = 101
        self.max_string_len = 20
        self.embedding_dim = 100
        self.rnn_hidden_size = 100
        self.random_seed = 2524
        self.bidirectional = True
        self.dropout_rate = 0.2
        self.eval_every = 5000
        self.clip = 0.25
        self.num_layers = 3
        self.filter_count = 25
        self.filter_count2 = 25
        self.filter_count3 = 25
        self.codec = 'UTF-8'
        self.experiment_out_dir = "experiments"
        self.dataset_name = "dataset"
        self.model_name = "model"
        self.increasing = False
        if filename:
            self.__dict__.update(json.load(open(filename)))
        self.random = random.Random(self.random_seed)

    def to_json(self):
        res = {}
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is str or type(self.__dict__[k]) is float or type(self.__dict__[k]) is int:
                res[k] = self.__dict__[k]
        return json.dumps(res)

    def save_config(self,exp_dir):
        with open(os.path.join(exp_dir,"config.json"), 'w') as fout:
            fout.write(self.to_json())
            fout.write("\n")


DefaultConfig = Config()
