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

# main
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

CALIB_PATH = 'D:/git/RTMapsSW_AP/src/fcn/calib_param/171023/calib.p'
MODEL_DIR = 'D:/git/RTMapsSW_AP/src/fcn/python/model-pool5_20171012-20000'
IMG_DIR = 'E:/Logging/PRJ_AP/20171025_HYU/export/20171108_200203_RecFile_1'
OUT_DIR = os.path.join(IMG_DIR, 'output')

calib = pickle.load(open(CALIB_PATH,"rb"))
t_m = calib["rectified"]
img_size = (int(calib["img_size"][0][0]), int(calib["img_size"][0][1]))

print('[Available Computing Devices]')
print(device_lib.list_local_devices()) 

postfix=""
hypes = load_hypes_from_logdir(MODEL_DIR);
modules = {}

f = os.path.join("architecture.py")
arch = imp.load_source("arch_%s" % postfix, f)
print('[log] architecture.py is loaded')
modules['arch'] = arch

f = os.path.join("objective.py")
objective = imp.load_source("objective_%s" % postfix, f)
print('[log] objective.py is loaded')
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

vgg16_npy_path = 'D:/git/RTMapsSW_AP/src/fcn/python/weights/vgg16.npy'

# Create tf graph and build module.
with tf.Graph().as_default():
	# Create placeholder for input
	image_pl = tf.placeholder(tf.float32)
	image = tf.expand_dims(image_pl, 0)

	# build Tensorflow graph using the model from logdir
	logits = modules['arch'].inference(hypes, image, vgg16_npy_path, train=False)
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
				print(file + 'is skipped')
				continue
			image_file = os.path.join(IMG_DIR, file)
			print("Input image: {}".format(image_file))

			image = scp.misc.imread(image_file)
			img_c= image[20:459, 13:222, :]
			img_recti = cv2.warpPerspective(img_c, t_m, dsize=img_size)
			img_recti = cv2.cvtColor(cv2.flip(cv2.transpose(img_recti),0), cv2.COLOR_BGR2RGB)            
			img_recti[85:175,74:298]=np.uint8([0,0,0])

			# Run KittiSeg model on image
			start_time = time.time()
			feed = {image_pl: img_recti}
			softmax = prediction['softmax']

			output = sess.run([softmax], feed_dict=feed) # list, [51200,4]
			dt = (time.time() - start_time)

			print('Speed (msec): {}'.format(1000*dt))
			print('Speed (fps): {}'.format(1/dt))

            # Reshape output from flat vector to 2D Image
			shape = img_recti.shape
			img_sum = np.zeros_like(img_recti[:,:,0])

			for c in range(hypes['arch']['num_classes']):
				img_prob = output[0][:, c].reshape(shape[0], shape[1])
				threshold_name = 'threshold_{:d}'.format(c)
				threshold = np.array(hypes['threshold'][threshold_name])
				img_thresh = (img_prob > threshold)*(c+1)
				img_sum = img_sum + img_thresh 
				
				image_name = os.path.join(output_folder_labeled, file.split('.')[0] + '_{}.png'.format(c))
				scp.misc.imsave(image_name, img_thresh)
            
			overlay_seg = utils.overlay_segmentation(img_recti,img_sum,color_seg)
			overlay_image_name = os.path.join(output_folder_overlay, file.split('.')[0] + '_overlay.png')
			scp.misc.imsave(overlay_image_name, overlay_seg)
