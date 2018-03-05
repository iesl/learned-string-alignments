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

import torch
import codecs
import subprocess
import json


def file_lines(filename,codec):
    f = codecs.open(filename,'r',codec)
    for line in f:
        yield line.decode(codec)

    f.close()

def row_wise_dot(tensor1, tensor2):
    return torch.sum(tensor1 * tensor2, dim=1,keepdim=True)

def wc_minus_l(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def __filter_json(the_dict):
    print("__filter_json")
    print(the_dict)
    res = {}
    for k in the_dict.keys():
        print("k : {} \t {} \t {}".format(k,the_dict[k],type(the_dict[k])))
        if type(the_dict[k]) is str or type(the_dict[k]) is float or type(the_dict[k]) is int or type(the_dict[k]) is list:
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = __filter_json(the_dict[k])
    print("res: {} ".format(res))
    return res

def save_dict_to_json(the_dict,the_file):
    with open(the_file, 'w') as fout:
        fout.write(json.dumps(__filter_json(the_dict)))
        fout.write("\n")
