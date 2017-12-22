"""
Plots MaxF1 score.

-------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

More details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import sys

import matplotlib.pyplot as plt

runfolder = '/media/srg/Work/git/avm_dataset/dataset/AVM6414_augmented' #'D:/git/KittiSeg/RUNS' 
anafile = 'output.log'

output_folder = os.path.join(runfolder, 'analyse')
if os.path.isdir(output_folder) == False:
    os.mkdir(output_folder)

filename = os.path.join(runfolder, anafile)
begin_iter = 55000
eval_iters = 5000
max_iters = 95000


def read_values(class_num, prop):
    regex_string = "\[class\s%d\]\s%s\:\s+(\d+\.\d+)" % (class_num, prop)    
    regex = re.compile(regex_string)

    value_list = [regex.search(line).group(1) for line in open(filename)
                  if regex.search(line) is not None]

    float_list = [float(value) for value in value_list]

    return float_list

def plot_training_result(prop, begin_iter, unit_iter, max_iter, n):
    label_list = range(begin_iter, max_iter+1, unit_iter)
    
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14})
    
    class_num = range(0,n)
    for c in class_num:
        values = read_values(c, prop)
        max_value = max(values)
        max_iter = values.index(max(values))* unit_iter + begin_iter
        plt.plot(label_list, values, label='class {}'.format(c), marker=".", linestyle='-')
        plt.text(65000, 35-c*5, '[calss {}] iter: {}, val: {}'.format(c, max_iter, max_value), size='x-small')        
    plt.xlabel('Iteration')
    plt.ylim([0,100])
    plt.ylabel('{} [%]'.format(prop))
    plt.legend(loc=0)
    
    plt.savefig(output_folder + "/{}".format(prop) + ".pdf")
    plt.show()

# MaxF1
class_num = 4
prop = 'MaxF1'
plot_training_result(prop, begin_iter, eval_iters, max_iters, class_num)

# Average Precision
prop = 'Average Precision'
plot_training_result(prop, begin_iter, eval_iters, max_iters, class_num)
