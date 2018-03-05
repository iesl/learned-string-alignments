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

class Vocab(object):
    def __init__(self,filename,max_string_len):
        self.filename = filename
        self.item2id,self.id2item = self.load(self.filename)
        self.OOV = "<OOV>"
        self.OOV_INDEX = 1
        self.PADDING_INDEX = 0
        self.item2id[self.OOV] = self.OOV_INDEX
        self.id2item[self.OOV_INDEX] = self.OOV
        self.max_string_len = int(max_string_len)
        self.size = len(self.item2id)

    def __len__(self):
        return self.size

    def load(self,filename):
        print("Loading vocab {}".format(filename))
        item2id = dict()
        id2item = dict()
        with codecs.open(filename,'r','UTF-8') as fin:
            for line in fin:
                splt = line.split("\t")
                item = splt[0]
                id = int(splt[1].strip())
                item2id[item] = id
                id2item[id] = item
        return item2id,id2item

    def to_ints(self,string):
        arr = []
        for c in list(string):
            arr.append(self.item2id.get(c,self.OOV_INDEX))
        if len(arr) > self.max_string_len:
            return np.asarray(arr[0:self.max_string_len])
        while len(arr) < self.max_string_len:
            arr += [0]
        return np.asarray(arr)

    def to_ints_no_pad(self,string):
        arr = []
        for c in list(string):
            arr.append(self.item2id.get(c,self.OOV_INDEX))
        if len(arr) > self.max_string_len:
            return np.asarray(arr[0:self.max_string_len])
        return np.asarray(arr)

    def to_string(self,list_ints):
        stri = ""
        for c in list_ints:
            if c != self.PADDING_INDEX:
                stri += self.id2item.get(c,self.OOV).encode("utf-8")
        return stri
