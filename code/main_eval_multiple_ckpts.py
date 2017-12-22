import os
import sys

for iter in range(75000,95001,5000):
    cmd = 'python evaluate.py --model-dir=/media/srg/Work/git/KittiSeg/RUNS/AVM_20171120/model_files --model-ckpt=/media/srg/Work/git/KittiSeg/RUNS/AVM_20171120/model.ckpt-{} --eval-file=/media/srg/Work/git/avm_dataset/dataset/AVM6414_augmented/test.txt --eval-dir=/media/srg/Work/git/avm_dataset/dataset/AVM6414_augmented'.format(iter)
    #print(cmd)
    os.system(cmd)
    

