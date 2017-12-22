import os
import sys

for f in range(4,10):
    cmd = "python runnable.py --model-dir=./models/model_170826 --img-dir=E:/ParkingSlotDetection/db/[20171222]_LG_AVM/rectified/set{} --output-dir=E:/ParkingSlotDetection/db/[20171222]_LG_AVM/ss_model_170826/set{} --save-labeled True".format(f, f)
    os.system(cmd)