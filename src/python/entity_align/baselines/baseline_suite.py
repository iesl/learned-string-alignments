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
import numpy as np
from operator import itemgetter
import argparse
import jellyfish
import fuzzy
from fuzzywuzzy import fuzz
import os

import sys

#Contains suite of baselines.

from entity_align.eval.EvalMap import eval_map_file
from entity_align.eval.EvalHitsAtK import eval_hits_at_k_file

import itertools

def lcs_length(xs, ys):
    '''Return the length of the LCS of xs and ys.

    Example:
    >>> lcs_length("HUMAN", "CHIMPANZEE")
    4
    '''
    ny = len(ys)
    curr = list(itertools.repeat(0, 1 + ny))
    for x in xs:
        prev = list(curr)
        for i, y in enumerate(ys):
            if x == y:
                curr[i+1] = prev[i] + 1
            else:
                curr[i+1] = max(curr[i], prev[i+1])
    return curr[ny]

#https://web.archive.org/web/20100602093104/http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
def mineditdist(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def dynamic_batcher(sources, start_index):
    current_token = sources[start_index]
    i = 0
    while sources[start_index + i] == current_token and len(sources) > start_index + i:
        i += 1
    end_index = start_index + i
    return end_index, end_index - start_index

def dynamic_k(labels):
    return sum(labels)

def baseline_write(data, labels, dist_fn,outfilename):
    """ Make predictions and write them to a file.
    :param data: string matrix of size [data_size, 2] where the 2 columns are the two strings being compared
    :param labels: Ground truth (whether or not these two strings refer to the same entity)
    :param dist_fn: name of distance function being used.
        "jaro": Jaro distance.
        "jaro_winkler": Jaro-Winkler distance.
        "LCS": Longest common subsequence.
        "fuzzytokensort": Token-level string edit distance.
        Anything else: String edit distance.
    :param outfilename: Name of file to write to.
    :param neg_len: lengths of negatives
    :return:
    """
    distances = []
    if dist_fn=="jaro":
        for i in range(0, len(data)):
            if i % 10000 == 0:
                print "{} data processed".format(i)
            distances.append(jellyfish.jaro_distance(data[i][0], data[i][1]))
    elif dist_fn=="jaro_winkler":
        for i in range(0, len(data)):
            if i % 10000 == 0:
                print "{} data processed".format(i)
            distances.append(jellyfish.jaro_winkler(data[i][0], data[i][1]))
    elif dist_fn=="lcs":
        for i in range(0, len(data)):
            if i % 10000 == 0:
                print "{} data processed".format(i)
            length = lcs_length(data[i][0], data[i][1])
            distances.append(length)
    elif dist_fn=="fuzzytokensort":
        for i in range(0, len(data)):
            if i % 10000 == 0:
                print "{} data processed".format(i)
            length = fuzz.token_set_ratio(data[i][0], data[i][1])
            distances.append(length)
    else:
        for i in range(0, len(data)):
            if i % 10000 == 0:
                print "{} data processed".format(i)
            distances.append(-1 * mineditdist(data[i][0], data[i][1]))
    with codecs.open(outfilename,'w+','UTF-8') as fout:
        for i in range(0, len(data)):
            fout.write(u"{}\t{}\t{}\t{}\n".format(data[i][0].decode("UTF-8"),data[i][1].decode("UTF-8"),labels[i],distances[i]))
        fout.flush()
        fout.close()



def tokens_to_vocab(data, phonetic_encoding="none"):
    """
    Used for token level string edit distance or phonetic encoding.
    Transforms tokens to vocabulary (t1 = a, t2 = b, t3 = c...)
    :param data: matrix, (data_size by 3) one line is [s1 \t s2 \t label], where s1 is made of tokens t1 t2 t3...
    :param labels: Ground truth (whether or not these two strings refer to the same entity)
    :param phonetic_encoding: name of phonetic encoding for string being used.
        "soundex": Soundex encoding
        "nysiis": Nysiis encoding
        "LCS": Longest common subsequence.
        Anything else: Program will shut down.
    :Returns array of size
    :return: (data_size by 3) where each line is s1 \t s2 \t label where s1/2 are encoded appropriately.
    """

    new_data = []
    if phonetic_encoding == "soundex":
        soundex = fuzzy.Soundex(4)
    ct = 0
    for line in data:
        ct += 1
        if ct % 10000 == 0:
            print("{} tokenized!".format(ct))
        tokens_made = []
        new_strings = ["", ""]
        new_string_1 = ""
        new_string_2 = ""
        for idx in [0, 1]:
            phrase = line.rstrip()[idx]
            for token in phrase.split(" "):
                #PHONETIC ENCODING WILL ONLY WORK WITH STRING EDIT DISTANCE
                if phonetic_encoding == "nysiis":
                    token = fuzzy.nysiis(token)
                elif phonetic_encoding == "soundex":
                    token = token.encode("UTF-8")
                    try:
                        token = soundex(token)
                    except:
                        print("Could not apply soundex to {}".format(token))
                        token = token
                else:
                    print("Provide phonetic encoding")
                    sys.exit(1)
                if token not in tokens_made:
                    tokens_made.append(token)
                if len(tokens_made) > 100:
                    print("TOO LONG!", tokens_made)
                new_strings[idx] += chr(ord("!") + tokens_made.index(token))
                new_strings[idx] = new_strings[idx].encode("UTF-8")
        new_data.append(new_strings)
    return new_data

def predict_baseline_results(output_dir, prediction_filename,dist_fn, phonetic_encoding):
    """
    Makes predictions given output file
    :param data: matrix, (data_size by 3) one line is [s1 \t s2 \t label], where s1 is made of tokens t1 t2 t3...
    :param labels: Ground truth (whether or not these two strings refer to the same entity)
    :param phonetic_encoding: name of phonetic encoding for string being used.
        "soundex": Soundex encoding
        "nysiis": Nysiis encoding
        "LCS": Longest common subsequence.
        Anything else: Program will shut down.
    :Returns array of size
    :return: (data_size by 3) where each line is s1 \t s2 \t label where s1/2 are encoded appropriately.
    """
    scores = ""
    map_score = eval_map_file(prediction_filename)
    scores += "{}\tMAP\t{}\n".format(dist_fn,map_score)
    scores += "{}\tHits@1\t{}\n".format(dist_fn,eval_hits_at_k_file(prediction_filename,1))
    scores += "{}\tHits@10\t{}\n".format(dist_fn,eval_hits_at_k_file(prediction_filename,10))
    scores += "{}\tHits@50\t{}\n".format(dist_fn,eval_hits_at_k_file(prediction_filename,50))
    print(scores)
    with open(os.path.join(output_dir,'{}.phonetic_{}.scores.tsv'.format(dist_fn, phonetic_encoding)),'w') as fout:
        fout.write(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--dist_fn", default="sed")
    parser.add_argument("--phonetic_encoding", default = "none")
    parser.add_argument("--token_level", default = "0")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    dev_file = args.dev_file
    dataset = args.dataset
    output_dir = args.output_dir
    phonetic_encoding = args.phonetic_encoding
    token_level = True if args.token_level == "1" else False
    dist_fn = args.dist_fn
    dev_data = []
    dev_labels = []
    print("loading data")
    with codecs.open(dev_file, "r", "UTF-8") as devf:
        for line in devf:
            line = line.strip()
            split = line.split("\t")
            dev_data.append([split[0], split[1]])
            dev_labels.append(int(split[2]))
        devf.close()
    print("data loaded")
    if token_level == True:
        print("beginning tokenization")
        dev_data = tokens_to_vocab(dev_data, phonetic_encoding = phonetic_encoding)
        print("done tokenizing")
    wf = os.path.join(output_dir,'{}.phonetic_{}_{}.output.tsv'.format(dist_fn, phonetic_encoding, dataset))
    baseline_write(dev_data, dev_labels, dist_fn, wf)
    predict_baseline_results(output_dir, wf, dist_fn, phonetic_encoding)
