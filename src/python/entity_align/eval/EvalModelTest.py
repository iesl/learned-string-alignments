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
import os
import torch

from entity_align.eval.Predict import write_predictions
from entity_align.model.Vocab import Vocab
from entity_align.utils.Config import Config
from entity_align.utils.DevTestBatcher import TestBatcher
from entity_align.eval.EvalHitsAtK import eval_hits_at_k_file
from entity_align.eval.EvalMap import eval_map_file
from entity_align.utils.Util import save_dict_to_json

if __name__ == "__main__":
    configfile = sys.argv[1]
    modelfile = sys.argv[2]

    config = Config(configfile)
    vocab = Vocab(config.vocab_file,config.max_string_len)
    model = torch.load(modelfile).cuda()
    test_batcher = TestBatcher(config,vocab)
    prediction_filename = os.path.join(config.experiment_out_dir,"test.predictions")
    write_predictions(model,test_batcher,prediction_filename)

    # score
    scores = ""
    map_score = float(eval_map_file(prediction_filename))
    hits_at_1 = float(eval_hits_at_k_file(prediction_filename, 1))
    hits_at_10 = float(eval_hits_at_k_file(prediction_filename, 10))
    hits_at_50 = float(eval_hits_at_k_file(prediction_filename, 50))
    scores += "{}\t{}\t{}\tMAP\t{}\n".format(config.model_name, config.dataset_name, "TEST", map_score)
    scores += "{}\t{}\t{}\tHits@1\t{}\n".format(config.model_name, config.dataset_name, "TEST", hits_at_1)
    scores += "{}\t{}\t{}\tHits@10\t{}\n".format(config.model_name, config.dataset_name, "TEST", hits_at_10)
    scores += "{}\t{}\t{}\tHits@50\t{}\n".format(config.model_name, config.dataset_name, "TEST", hits_at_50)
    print(scores)
    score_obj = {"map": map_score, "hits_at_1": hits_at_1, "hits_at_10": hits_at_10,
                 "hits_at_50": hits_at_50,
                 "config": config.__dict__}
    print(score_obj)
    save_dict_to_json(score_obj, os.path.join(config.experiment_out_dir, 'test.scores.json'))
    with open(os.path.join(config.experiment_out_dir, 'test.scores.tsv'), 'w') as fout:
        fout.write(scores)
