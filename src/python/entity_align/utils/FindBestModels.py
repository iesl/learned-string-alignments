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
import json
import os
from collections import defaultdict
from entity_align.utils.Config import Config

if __name__ == "__main__":
    file_of_scores = sys.argv[1]
    score_objs = []
    with open(file_of_scores, 'r') as fin:
        for line in fin:
            js = json.loads(line.strip())
            c = Config()
            c.__dict__ = js['config']
            js['config'] = c
            score_objs.append(js)
    by_model_by_dataset = defaultdict(dict)
    for js in score_objs:
        if js['config'].model_name not in by_model_by_dataset \
                or js['config'].dataset_name not in by_model_by_dataset[js['config'].model_name] \
                or by_model_by_dataset[js['config'].model_name][js['config'].dataset_name]["map"] < js["map"]:
            by_model_by_dataset[js['config'].model_name][js['config'].dataset_name] = js

    print("=======================")
    print("=======================")
    print("Best Model Scores")
    for model in by_model_by_dataset.keys():
        for ds in by_model_by_dataset[model].keys():
            js = by_model_by_dataset[model][ds]
            print("{}\t{}\t{}\t{}".format(js['config'].model_name, js['config'].dataset_name, "MAP", js['map']))
            print(
                "{}\t{}\t{}\t{}".format(js['config'].model_name, js['config'].dataset_name, "HITS@1", js['hits_at_1']))
            print("{}\t{}\t{}\t{}".format(js['config'].model_name, js['config'].dataset_name, "HITS@10",
                                          js['hits_at_10']))
            print("{}\t{}\t{}\t{}".format(js['config'].model_name, js['config'].dataset_name, "HITS@50",
                                          js['hits_at_50']))

    print("=======================")
    print("=======================")

    print()
    print()
    print("=======================")
    print("=======================")
    print("Commands to run Eval")
    for model in by_model_by_dataset.keys():
        for ds in by_model_by_dataset[model].keys():
            js = by_model_by_dataset[model][ds]
            exp_dir = js['config'].experiment_out_dir
            samples = js['samples']
            config_file = os.path.join(exp_dir,"config.json")
            model_file = os.path.join(exp_dir,"model_{}_{}_{}.torch".format(model,ds,samples))
            print("sh bin/eval/run_eval.sh {} {} ADD_GPU_ID_HERE".format(config_file,model_file))
    print("=======================")
    print("=======================")
