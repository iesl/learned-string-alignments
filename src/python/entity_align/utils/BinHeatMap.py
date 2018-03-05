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


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Create binary heat map used in poster.    
    """

    str1 = "Bill Hicks"
    str2 = "Hicks, William Melvin"
    scores = np.ones((len(str1),len(str2)))
    for i in range(len(str1)):
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                scores[i,j] = 0
    fig = plt.figure()
    max_interesting = 0
    my_yticks = list(str1)
    plt.yticks(range(len(str1)), my_yticks)
    my_xticks = list(str2)
    plt.xticks(range(len(str2)) , my_xticks)
    plt.imshow(scores, cmap='hot', interpolation='nearest')
    plt.gca().invert_yaxis()
    fig.savefig("binary.pdf")
