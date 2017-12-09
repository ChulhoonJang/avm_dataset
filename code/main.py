import os
import sys

for f in range(1,19):
    cmd = "python runnable.py --model-dir=./models/model_171115 --img-dir=D:/tmpLogging/AP/171130_HYU/export/rectified/set{} --output-dir=D:/tmpLogging/AP/171130_HYU/export/ss/set{} --save-labeled True".format(f, f)
    os.system(cmd)