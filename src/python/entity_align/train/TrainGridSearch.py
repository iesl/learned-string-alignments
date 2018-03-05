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

from entity_align.train.TrainModel import train_model
from entity_align.utils.GridSearchConfig import GridSearchConfig

if __name__ == "__main__":

    # Set up the config
    grid_search_config = GridSearchConfig(sys.argv[1])
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    for config in grid_search_config.configs_iter():
        print("Running On Config:")
        print(config.__dict__)
        train_model(config,dataset_name,model_name)
