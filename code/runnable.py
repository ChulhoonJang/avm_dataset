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

def get_arguments():
    parser = argparse.ArgumentParser(description="Fully Convolution Network - KittiSeg")
    
    parser.add_argument("--model-dir", type=str,
                        help="Directory of the trained model", required=True)
    parser.add_argument("--img-dir", type=str,
                        help="Directory of the image folder", required=True)
    parser.add_argument("--vgg16", type=str, default='./weights/vgg16.npy',
                        help="VGG16 params")
    parser.add_argument("--save-labeled", type=bool, default=False,
                        help="To save seperated labeled images")
    return parser.parse_args()


def main():
    """Create the model and start the training."""
    
    # argument parsing
    args = get_arguments()
    
    MODEL_DIR = args.model_dir
    IMG_DIR = args.img_dir
    VGG16_PATH = args.vgg16
    OUT_DIR = os.path.join(IMG_DIR, 'output')
    
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

    if os.path.isdir(OUT_DIR) == False:
        os.mkdir(OUT_DIR)
        
    output_folder_overlay = os.path.join(OUT_DIR, 'overlay')
    output_folder_labeled = os.path.join(OUT_DIR, 'labeled')
    output_folder_labeled_raw = os.path.join(OUT_DIR, 'labeled_raw')

    if os.path.isdir(output_folder_overlay) == False:
        os.mkdir(output_folder_overlay)
    if os.path.isdir(output_folder_labeled) == False:
        os.mkdir(output_folder_labeled)
    if os.path.isdir(output_folder_labeled_raw) == False :
        os.mkdir(output_folder_labeled_raw)

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

            # Load and resize input image
            for file in os.listdir(IMG_DIR):
                if len(file.split('.')) != 2:
                    continue
                if file.split('.')[1] != 'jpg':
                    print(file + ' is skipped')
                    continue
                image_file = os.path.join(IMG_DIR, file)
                print("Input image: {}".format(image_file))

                img = scp.misc.imread(image_file)

                # Run KittiSeg model on image
                start_time = time.time()
                feed = {image_pl: img}
                softmax = prediction['softmax']

                output = sess.run([softmax], feed_dict=feed) # list, [51200,4]
                dt = (time.time() - start_time)

                print('Speed (msec): {}'.format(1000*dt))
                print('Speed (fps): {}'.format(1/dt))

                # Reshape output from flat vector to 2D Image
                shape = img.shape
                img_sum = np.zeros_like(img[:,:,0])

                for c in range(hypes['arch']['num_classes']):
                    img_prob = output[0][:, c].reshape(shape[0], shape[1])
                    threshold_name = 'threshold_{:d}'.format(c)
                    threshold = np.array(hypes['threshold'][threshold_name])
                    img_thresh = (img_prob > threshold)*(c+1)
                    img_sum = img_sum + img_thresh 
                    if args.save_labeled == True:
                        image_name = os.path.join(output_folder_labeled, file.split('.')[0] + '_{}.png'.format(c))
                        scp.misc.imsave(image_name, img_thresh)
                
                overlay_seg = utils.overlay_segmentation(img,img_sum,color_seg)
                overlay_image_name = os.path.join(output_folder_overlay, file.split('.')[0] + '.png')
                scp.misc.imsave(overlay_image_name, overlay_seg)

    
if __name__ == '__main__':
    main()
