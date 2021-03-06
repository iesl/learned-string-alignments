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
import codecs

class DevTestBatcher(object):
    """
        Class for Dev/Test batching
    """

    def __init__(self, config, vocab, use_dev):
        """Construct a DevTestBatcher

            Construct a batcher that works on the Dev / test set
        """
        self.config = config
        self.vocab = vocab
        self.use_dev = use_dev
        if use_dev:
            self.filename = config.dev_file
        else:
            self.filename = config.test_file
        self.batch_size = self.config.dev_batch_size

    def batches(self):
        """Provide all batches in the dev/test set

        Generator over batches in the dataset. Note that the last batch may be
         of a different size than the other batches

        :return: Generator over bathes of size self.config.dev_batch_size.
            Each element of the generator contains the following tuple:
                batch_queries,
                batch_query_lengths,
                batch_query_strings,
                batch_targets,
                batch_target_lengths,
                batch_target_strings,
                batch_labels,
                batch_size
        """
        batch_queries = []
        batch_query_lengths = []
        batch_query_strings = []
        batch_targets = []
        batch_target_lengths = []
        batch_target_strings = []
        batch_labels = []
        counter = 0
        with codecs.open(self.filename,'r','UTF-8') as fin:
            for line in fin:
                if counter % self.batch_size == 0 and counter > 0:
                    yield np.asarray(batch_queries),\
                          np.asarray(batch_query_lengths),\
                          batch_query_strings,\
                          np.asarray(batch_targets),\
                          np.asarray(batch_target_lengths),\
                          batch_target_strings,\
                          np.asarray(batch_labels),\
                          self.batch_size
                    batch_queries = []
                    batch_query_lengths = []
                    batch_query_strings = []
                    batch_targets = []
                    batch_target_lengths = []
                    batch_target_strings = []
                    batch_labels = []


                split = line.rstrip().split("\t")

                # Note that the vocabulary knows about the max_string_length
                # which is why we do not need take it into account here.
                if len(split) < 3:
                    print(split)
                query_string = split[0]
                query_len = [min(self.config.max_string_len,len(query_string)) - 1]
                query_vec = np.asarray(self.vocab.to_ints(query_string))

                target_string = split[1]
                target_len = [min(self.config.max_string_len,len(target_string)) - 1]
                target_vec = np.asarray(self.vocab.to_ints(target_string))

                label = int(split[2])

                batch_queries.append(query_vec)
                batch_query_lengths.append(query_len)
                batch_query_strings.append(query_string)

                batch_targets.append(target_vec)
                batch_target_lengths.append(target_len)
                batch_target_strings.append(target_string)

                batch_labels.append(label)

                counter += 1
        if len(batch_queries) >= 1:
            yield np.asarray(batch_queries), \
                  np.asarray(batch_query_lengths), \
                  batch_query_strings, \
                  np.asarray(batch_targets), \
                  np.asarray(batch_target_lengths), \
                  batch_target_strings, \
                  np.asarray(batch_labels), \
                  len(batch_queries)

class DevBatcher(DevTestBatcher):
    def __init__(self,config,vocab):
        super(self.__class__, self).__init__(config,vocab,use_dev=True)

class TestBatcher(DevTestBatcher):
    def __init__(self,config,vocab):
        super(self.__class__, self).__init__(config,vocab,use_dev=False)
