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

def count_chars(inputfile, outputfile, minCount):
    char_dict = {}
    counter = 0
    with codecs.open(inputfile, "r", "UTF-8", errors= "ignore") as rf:
        for line in rf:
            splt = line.strip().split("\t")
            if counter % 1000 == 0:
                sys.stdout.write("\rProcessed {} lines".format(counter))
            for s in splt:
                for char in s:
                    if char not in char_dict:
                        char_dict[char] = 1
                    else:
                        char_dict[char] += 1
            counter += 1
    sys.stdout.write("\nDone....Now Writing Vocab.")
    with codecs.open(outputfile, "w+", "UTF-8") as wf:
        charid = 2
        for char in char_dict.keys():
            if char_dict[char] >= minCount:
                wf.write("{}\t{}\n".format(char,charid))
                charid += 1
        wf.flush()
        wf.close()

if __name__ == "__main__":
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    minCount = int(sys.argv[3])
    count_chars(inputfile, outputfile, minCount)
