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

import datetime
import os
import sys
from shutil import copyfile, copytree

import torch
import torch.optim as optim

from entity_align.eval.EvalHitsAtK import eval_hits_at_k_file
from entity_align.eval.EvalMap import eval_map_file
from entity_align.eval.Predict import write_predictions
from entity_align.model.AlignCNN import AlignCNN
from entity_align.model.AlignBinary import AlignBinary
from entity_align.model.AlignDot import AlignDot
from entity_align.model.Vocab import Vocab
from entity_align.model.AlignLinear import AlignLinear
from entity_align.utils.Batcher import Batcher
from entity_align.utils.Config import Config
from entity_align.utils.DevTestBatcher import DevBatcher
from entity_align.utils.Util import save_dict_to_json

def train_model(config,dataset_name,model_name):
    """ Train based on the given config, model / dataset
    
    :param config: config object
    :param dataset_name: name of dataset
    :param model_name: name of model
    :return: 
    """
    config.dataset_name = dataset_name
    now = datetime.datetime.now()
    config.model_name = model_name
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    config.experiment_out_dir = os.path.join("exp_out", dataset_name, model_name, ts)

    # Load vocab
    vocab = Vocab(config.vocab_file, config.max_string_len)

    # Set up output dir
    output_dir = config.experiment_out_dir
    os.makedirs(output_dir)

    # save the config to outdir
    config.save_config(output_dir)
    # save the vocab to out dir
    copyfile(config.vocab_file, os.path.join(output_dir, 'vocab.tsv'))
    # save the source code.
    copytree(os.path.join(os.environ['SED_ROOT'], 'src'), os.path.join(output_dir, 'src'))

    torch.manual_seed(config.random_seed)

    # Set up batcher
    batcher = Batcher(config, vocab, 'train')

    model = None
    # Set up Model
    if config.model_name == "AlignCNN":
        model = AlignCNN(config, vocab)
    elif config.model_name == "AlignDot":
        model = AlignDot(config, vocab)
    elif config.model_name == "AlignBinary":
        model = AlignBinary(config, vocab)
    elif config.model_name == "AlignLinear":
        model = AlignLinear(config, vocab)
    else:
        print("Unknown model")
        sys.exit(1)

    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                           weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    counter = 0
    sum_loss = 0.0

    print("Begin Training")
    sys.stdout.flush()

    # Training loop
    for source, pos, neg, source_len, pos_len, neg_len in batcher.get_next_batch():
        counter = counter + 1
        optimizer.zero_grad()

        loss = model.compute_loss(source, pos, neg, source_len, pos_len, neg_len)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        if counter % 100 == 0:
            # print("p-n:{}".format(model.print_loss(source,pos,neg,source_len,pos_len,neg_len)))
            this_loss = loss.cpu().data.numpy()[0]
            sum_loss += this_loss
            print("Processed {} batches, Loss of batch {}: {}. Average loss: {}".format(counter, counter, this_loss,
                                                                                        sum_loss / (counter / 100)))
            sys.stdout.flush()

        if counter % config.eval_every == 0:
            dev_batcher = DevBatcher(config, vocab)
            prediction_filename = os.path.join(output_dir, 'dev.predictions.{}.tsv').format(counter)
            write_predictions(model, dev_batcher, prediction_filename)
            scores = ""
            map_score = float(eval_map_file(prediction_filename))
            hits_at_1 = float(eval_hits_at_k_file(prediction_filename, 1))
            hits_at_10 = float(eval_hits_at_k_file(prediction_filename, 10))
            hits_at_50 = float(eval_hits_at_k_file(prediction_filename, 50))
            scores += "{}\t{}\t{}\tMAP\t{}\n".format(config.model_name, config.dataset_name, counter, map_score)
            scores += "{}\t{}\t{}\tHits@1\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_1)
            scores += "{}\t{}\t{}\tHits@10\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_10)
            scores += "{}\t{}\t{}\tHits@50\t{}\n".format(config.model_name, config.dataset_name, counter, hits_at_50)
            print(scores)
            score_obj = {"samples": counter, "map": map_score, "hits_at_1": hits_at_1, "hits_at_10": hits_at_10, "hits_at_50": hits_at_50,
                         "config": config.__dict__}
            print(score_obj)
            save_dict_to_json(score_obj, os.path.join(output_dir, 'dev.scores.{}.json'.format(counter)))
            with open(os.path.join(output_dir, 'dev.scores.{}.tsv'.format(counter)), 'w') as fout:
                fout.write(scores)
            if map_score > best_map:
                print("New best MAP!")
                print("Saving Model.....")
                torch.save(model, os.path.join(output_dir,
                                               'model_{}_{}_{}.torch'.format(config.model_name, config.dataset_name,
                                                                             counter)))
                best_map = map_score
            sys.stdout.flush()
        if counter == config.num_minibatches:
            break

if __name__ == "__main__":

    # Set up the config
    config = Config(sys.argv[1])
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    train_model(config,dataset_name,model_name)
