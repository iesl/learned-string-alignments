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

import sys

from entity_align.eval.EvalHitsAtK import eval_hits_at_k_file

from entity_align.eval.EvalMap import eval_map_file


def score(prediction_filename,model_name,dataset_name):
    """ Given a file of predictions, compute all metrics
    
    :param prediction_filename: TSV file of predictions
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :return: 
    """
    counter = 0
    scores = ""
    map_score = eval_map_file(prediction_filename)
    scores += "{}\t{}\t{}\tMAP\t{}\n".format(model_name, dataset_name, counter, map_score)
    scores += "{}\t{}\t{}\tHits@1\t{}\n".format(model_name, dataset_name, counter,
                                                eval_hits_at_k_file(prediction_filename, 1))
    scores += "{}\t{}\t{}\tHits@10\t{}\n".format(model_name, dataset_name, counter,
                                                 eval_hits_at_k_file(prediction_filename, 10))
    scores += "{}\t{}\t{}\tHits@50\t{}\n".format(model_name, dataset_name, counter,
                                                 eval_hits_at_k_file(prediction_filename, 50))
    return scores

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 2 else "model"
    dataset = sys.argv[4]if len(sys.argv) > 3 else "dataset"
    with open(out_file,'w') as fout:
        s = score(in_file,model,dataset)
        fout.write(s)
        print(s)
