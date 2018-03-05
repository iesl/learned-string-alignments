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

import codecs
import sys

def write_predictions(model,batcher,outfilename):
    """ Use the model to make predictions on the data in the batcher
    
    :param model: Model to use to score string alignments
    :param batcher: Batcher containing data to evaluate (a DevTestBatcher)
    :param outfilename: Where to write the predictions to a file for evaluation (tsv) (overwrites)
    :return: 
    """
    with codecs.open(outfilename,'w','UTF-8') as fout:
        for idx,batch in enumerate(batcher.batches()):
            if idx % 100 == 0:
                print("Predicted {} batches".format(idx))
                sys.stdout.flush()

            batch_queries,batch_query_lengths,batch_query_strings,\
            batch_targets,batch_target_lengths,batch_target_strings,\
            batch_labels,batch_size = batch

            scores = model.score_dev_test_batch(batch_queries,
                                                batch_query_lengths,
                                                batch_targets,
                                                batch_target_lengths,
                                                batch_size)
            if type(batch_labels) is not list:
                batch_labels = batch_labels.tolist()

            if type(scores) is not list:
                scores = list(scores.cpu().data.numpy().squeeze())

            for src,tgt,lbl,score in zip(batch_query_strings,batch_target_strings,batch_labels,scores):
                fout.write(u"{}\t{}\t{}\t{}\n".format(src,tgt,lbl,score))
