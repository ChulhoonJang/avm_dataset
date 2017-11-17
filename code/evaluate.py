from __future__ import print_function

import os
import imp
import json
import sys
import time
import scipy as scp
import tensorflow as tf
import numpy as np
import utils
import cv2
import pickle
import logging
import logging.handlers

from tqdm import tqdm
from tensorflow.python.client import device_lib

import argparse

def _add_paths_to_sys(hypes):
    """
    Add all module dirs to syspath.

    This adds the dirname of all modules to path.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    base_path = hypes['dirs']['base_path']
    if 'path' in hypes:
            for path in hypes['path']:
                path = os.path.realpath(os.path.join(base_path, path))
                sys.path.insert(1, path)
    return

def load_hypes_from_logdir(logdir):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    hypes
    """
    hypes_fname = os.path.join(logdir, "hypes.json")
    with open(hypes_fname, 'r') as f:
        hypes = json.load(f)

    _add_paths_to_sys(hypes)

    return hypes

def load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver

    Returns
    -----------
    int
        training step of checkpoint
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])

def eval_image(hypes, gt_image, inferred_image, num_class):
    """."""
    thresh = np.array(range(0, 256))/255.0

    gt_image_for_class=[]
    valid_gt = np.zeros(shape=(gt_image.shape[0],gt_image.shape[1]))
    
    for c in range(hypes['arch']['num_classes']):
    	# get class color code from hypes
        color_name = 'color_class_{:d}'.format(c)
        color_class = np.array(hypes['data'][color_name])
        
        # get true / false image according to the gt color code value
        gt = np.all(gt_image == color_class, axis=2)
        
        # save the binary image
        gt_image_for_class.append(gt)
		
		# set the valid area for evaluation
        valid_gt = valid_gt + gt


    thresInf = np.concatenate(([-np.Inf], thresh, [np.Inf])) # -inf, 0, ... , 1, inf
    
    #Merge validMap with validArea    
    validMap=valid_gt

    # histogram of false negatives
    fnArray = inferred_image[(gt_image_for_class[num_class] == True) & (validMap == True)]
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thresh)];

	# histogram of false positives
    fpArray = inferred_image[(gt_image_for_class[num_class] == False) & (validMap == True)]    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thresh)]

    # count labels and protos
    posNum = np.sum((gt_image_for_class[num_class] == True) & (validMap == True))
    negNum = np.sum((gt_image_for_class[num_class] == False) & (validMap == True))

    return FN, FP, posNum, negNum    

def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    
    return outDict
    
def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
	# calculate true positive
    totalTP = totalPosNum - totalFN # array
    # calculate true negative
    totalTN = totalNegNum - totalFP # array

    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    TNR    = totalTN / float( totalNegNum )
    
    precision =  totalTP / (totalTP + totalFP + 1e-10)
    accuracy = (totalTP + totalTN) / (float( totalPosNum ) + float( totalNegNum ))
    
    selector_invalid = (recall==0) & (precision==0)
    
    recall = recall[~selector_invalid] # array
    precision = precision[~selector_invalid] # array        
        
    #Pascal VOC average precision
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind == None:
            continue
        pmax = max(precision[ind])
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter    
    
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF= F[index]
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF
    prob_eval_scores['accuracy'] = accuracy

    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    prob_eval_scores['TNR'] = TNR
    prob_eval_scores['thresh'] = thresh
    
    if thresh is not None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh
        
    return prob_eval_scores
    
def get_arguments():
    parser = argparse.ArgumentParser(description="Fully Convolution Network - KittiSeg")
    
    parser.add_argument("--model-dir", type=str,
                        help="Directory of the trained model", required=True)
    parser.add_argument("--eval-file", type=str,
                        help="file for evaluation", required=True)
    parser.add_argument("--eval-dir", type=str,
                        help="Directory of the folder for evaluation", required=True)
    parser.add_argument("--vgg16", type=str, default='./weights/vgg16.npy',
                        help="VGG16 params")
    return parser.parse_args()


def main():
    """Create the model and start the training."""
    # argument parsing
    args = get_arguments()
    
    MODEL_DIR = args.model_dir
    EVAL_DIR = args.eval_dir
    EVAL_FILE = args.eval_file
    VGG16_PATH = args.vgg16

    # logger setting
    logger = logging.getLogger('Evaluation')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')

    fileHandler = logging.FileHandler('{}/output.log'.format(EVAL_DIR))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    #logger.addHandler(streamHandler)
    logger.setLevel(logging.DEBUG)
    
    logger.info('model: {}'.format(MODEL_DIR))
    logger.info('eval_file: {}'.format(EVAL_FILE))
    
    # initialization
    print('[Available Computing Devices]')
    print(device_lib.list_local_devices()) 

    postfix=""
    hypes = load_hypes_from_logdir(MODEL_DIR);
    modules = {}

    f = os.path.join("architecture.py")
    arch = imp.load_source("arch_%s" % postfix, f)
    modules['arch'] = arch

    f = os.path.join("objective.py")
    objective = imp.load_source("objective_%s" % postfix, f)
    modules['objective'] = objective

    color_seg={1:(0,153,255,127), 2:(0,255,0,127), 3:(255,0,0,127), 4:(255,0,255,127)}
		
	# create the evaluation variable
    eval_metrics = [] # for multi class evaluation        
    for c in range(hypes['arch']['num_classes']):
        metric = {}
        metric['thresh'] = np.array(range(0, 256))/255.0
        metric['total_fp'] = np.zeros(metric['thresh'].shape)
        metric['total_fn'] = np.zeros(metric['thresh'].shape)
        metric['total_posnum'] = 0
        metric['total_negnum'] = 0
        eval_metrics.append(metric)

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        logits = modules['arch'].inference(hypes, image, VGG16_PATH, train=False)
        prediction = modules['objective'].decoder(hypes, logits, train=False)

        print("Graph build successfully.")
        
        # Create a session for running Ops on the Graph.
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            saver = tf.train.Saver()

            # Load weights from logdir
            load_weights(MODEL_DIR, sess, saver)
            print('[log] wieghts are restored')
            print("Start inference")

			# open the image / gt list
            files = []
            with open(EVAL_FILE) as file:
                for i, datum in enumerate(file):
                    data = datum.rstrip() # training/images/00000100.jpg training/gt_images/00000100.png
                    image_file, gt_file = data.split(" ")
                    files.append(data.split(" "))
			
            for i in tqdm(range(len(files))):
                image_file = files[i][0]
                gt_file = files[i][1]
                image_file = os.path.join(EVAL_DIR, image_file)
                gt_file = os.path.join(EVAL_DIR, gt_file)

                # load the image and gt image
                image = scp.misc.imread(image_file, mode='RGB')
                gt_image = scp.misc.imread(gt_file, mode='RGB')
			
                # inference
                shape = image.shape
                feed_dict = {image_pl: image}
                softmax = prediction['softmax']
                
                output = sess.run([softmax], feed_dict=feed_dict)
			
                # evaluate
                for c in range(len(eval_metrics)):
                    output_im = output[0][:, c].reshape(shape[0], shape[1])
                    FN, FP, posNum, negNum = eval_image(hypes, gt_image, output_im, c)

                    eval_metrics[c]['total_fp'] += FP
                    eval_metrics[c]['total_fn'] += FN
                    eval_metrics[c]['total_posnum'] += posNum
                    eval_metrics[c]['total_negnum'] += negNum
	                
            eval_results = []
            for c in range(len(eval_metrics)):
                result = pxEval_maximizeFMeasure(eval_metrics[c]['total_posnum'], eval_metrics[c]['total_negnum'],
                                                 eval_metrics[c]['total_fn'], eval_metrics[c]['total_fp'], thresh=eval_metrics[c]['thresh'])
                eval_results.append(result)

            for c in range(len(eval_metrics)):
                logger.info('[class {:d}] MaxF1: {:0.4f}'.format(c, 100*eval_results[c]['MaxF']))
                logger.info('[class {:d}] BestThresh: {:0.4f}'.format(c, 100*eval_results[c]['BestThresh']))
                logger.info('[class {:d}] Average Precision: {:0.4f}'.format(c, 100*eval_results[c]['AvgPrec']))
    
if __name__ == '__main__':
    main()
