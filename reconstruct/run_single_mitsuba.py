import numpy as np
import OpenEXR
import Imath

import datetime
import os
import sys
import pdb
import time
import random
import collections
import concurrent.futures
import argparse

import config
import archive

from util import *
import image
import archive
import model

import imageio


### Command line options.
	
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=int, help='model index to use for the reconstruction')
parser.add_argument('--reconstruct', type=str, help='reconstruct the outputs of the given .xml file')
parser.add_argument('--run-metrics', type=str, help='run measurement metrics for the reconstruction against the given reference image.')
parser.add_argument('--no-benchmark', action="store_true", help='disables benchmarking execution time.')
parser.add_argument('--elpips-sample-count', type=int, default=200, help='number of samples for E-LPIPS measurement. Increase to lower stdev. Default: 200. Should probably be at least 50.')
args = parser.parse_args()
assert args.elpips_sample_count >= 50


# Validate the model configuration.
config.model.validate()


def readExr(file_path):
	exr_file = OpenEXR.InputFile(file_path)
	exr_header = exr_file.header()
	H, W = exr_header['dataWindow'].max.y + 1, exr_header['dataWindow'].max.x + 1
	assert all((exr_header['channels'][channel].type == Imath.PixelType(OpenEXR.HALF) for channel in ('R', 'G', 'B')))
	
	R, G, B = exr_file.channels(['R', 'G', 'B'])
	image_R = np.frombuffer(R, dtype=np.float16).reshape([H, W])
	image_G = np.frombuffer(G, dtype=np.float16).reshape([H, W])
	image_B = np.frombuffer(B, dtype=np.float16).reshape([H, W])
	image = np.stack([image_R, image_G, image_B], axis=2)
	
	return image

def saveExr(image, file_path):
	half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
	
	header = OpenEXR.Header(image.shape[1], image.shape[0])
	header['channels'] = dict(R=half, G=half, B=half)
	
	exr_file = OpenEXR.OutputFile(file_path, header)
	
	r = image[:, :, 0].astype(np.float16)
	g = image[:, :, 1].astype(np.float16)
	b = image[:, :, 2].astype(np.float16)
		
	exr_file.writePixels({
		'R': r.tobytes(),
		'G': g.tobytes(),
		'B': b.tobytes()
	})
	return image

	
def reconstruct_one(scene_xml):
	'''Reconstructs the given image.'''
	import tensorflow as tf

	directory, scene_xml = os.path.split(scene_xml)
	basename, ext = os.path.splitext(scene_xml)	
	
	def crop_edges(x):
		if config.PAD_WIDTH == 0:
			return x
		else:
			return x[config.PAD_WIDTH:-config.PAD_WIDTH, config.PAD_WIDTH:-config.PAD_WIDTH, :]


	# Create the model.
	input_channels, input_height, input_width = 3, 720, 1280
	input_width += 2 * config.PAD_WIDTH
	input_height += 2 * config.PAD_WIDTH
	
	# Build inputs.
	print("Creating GPU input.")

	with tf.device('/gpu:0'):
		with tf.name_scope('gpu0_input'):
			input_noisy = tf.Variable(tf.constant(0.0, shape=[1, config.INPUT_COUNT, input_height, input_width]), validate_shape=False, name='input_gpu0')
				
			input_clean = tf.Variable(tf.constant(0.0, shape=[1, config.TOTAL_COUNT - config.INPUT_COUNT, input_height, input_width]), validate_shape=False, name='input_clean_gpu0')

			gpu_inputs = {
				'input_noisy': input_noisy,
				'input_clean': input_clean
			}
	
	# Finalize the model.
	print("Constructing model... ", end="")
	stime = time.time()
	sys.stdout.flush()
	
	model_current = model.trainable_model(gpu_inputs, define_model_fn=model.reconstruct, compile_optimizer=False)
	
	
	etime = time.time()
	print("Done in {:.01f}s.".format(etime - stime))
			
	# Setting minibatch.
	assign_minibatch = model.assign_inference_input(gpu_inputs['input_noisy'])

	# Start TF session.
	tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	print("Creating TensorFlow session.")

	with tf.Session(config=tf_config) as sess:
		# Initialize the model.
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		print("**** Restoring model weights! ****")
		print("Loading weights for {}... ".format(model_current['name']), end="")
		sys.stdout.flush()
		model_current['saver'].restore(sess, os.path.join(config.SAVE_DIRECTORY, "{}.ckpt".format(model_current['name'])))
		print("Done.")
	
		tasks = collections.deque()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

			# Load the input images.
			
			def getBuffer(buffer):
				file_path = os.path.join(directory, basename + "-" + buffer + ".exr")
				return readExr(file_path)
	
			# Not implemented.
			assert not config.model.input_enabled('variances')
			assert not config.model.input_enabled('diffuse')
			assert not config.model.input_enabled('specular')
			
			# Read the required buffers.
			primal, dx, dy, albedo, normal, depth = None, None, None, None, None, None

			print("Loading images... ", end="")
			stime = time.time()
			sys.stdout.flush()
			
			primal = getBuffer("primal").astype(np.float32)
			primal = np.log(1.0 + np.maximum(0.0, primal))
			
			if config.model.input_enabled('gradients'):
				dx = getBuffer("dx").astype(np.float32)
				dx = np.sign(dx) * np.log(1.0 + np.abs(dx))
				
				dy = getBuffer("dy").astype(np.float32)				
				dy = np.sign(dy) * np.log(1.0 + np.abs(dy))
				
			if config.model.input_enabled('albedo'):
				albedo = getBuffer("albedo").astype(np.float32)
				
			if config.model.input_enabled('normal'):
				normal = getBuffer("normal").astype(np.float32)
				
			if config.model.input_enabled('depth'):
				depth = getBuffer("depth")[:, :, 0:1].astype(np.float32)
				depth /= 1e-10 + np.amax(depth)
	
			# Transpose all to NCHW.
			transpose = lambda x: np.transpose(x, [2, 0, 1]) if x is not None else None
		
			primal, dx, dy = transpose(primal), transpose(dx), transpose(dy)
			albedo, normal, depth = transpose(albedo), transpose(normal), transpose(depth)
				
			# Mirror-pad the buffers.
			pad = lambda buffer, f: f(buffer, (config.PAD_WIDTH, config.PAD_WIDTH), (config.PAD_WIDTH, config.PAD_WIDTH)) if buffer is not None else None
		
			buffers = []
			buffers.append(pad(primal, archive.mirrorPadPrimal))
			buffers.append(pad(dx, archive.mirrorPadDx))
			buffers.append(pad(dy, archive.mirrorPadDy))
			buffers.append(pad(albedo, archive.mirrorPadPrimal))
			buffers.append(pad(normal, archive.mirrorPadPrimal))
			buffers.append(pad(depth, archive.mirrorPadOdd))
			
			# Note: The order must be exactly the same as defined in config.py and archive.py.
		
			# Concatenate the enabled features.
			minibatch = [x for x in buffers if x is not None]
			minibatch = np.concatenate(minibatch, axis=0)
			minibatch = np.expand_dims(minibatch, axis=0)
	
			etime = time.time()
			print("Done in {:.01f}s.".format(etime - stime))
	
			
			# Send training data.
			for i in range(2):
				# Run twice to get a more accurate benchmark. TF still compiles the model on the first go.
				if args.no_benchmark and i > 0:
					break
					
				if i == 0:
					print("Compiling and running... ", end="")
				else:
					print("Re-running without compilation for benchmark... ", end="")
					
				sys.stdout.flush()
				stime = time.time()
				sess.run(assign_minibatch['kernel'], feed_dict={assign_minibatch['data']: minibatch})
							
				# Run reconstruction.
				reconstruction = sess.run(model_current[0]['y'])
				reconstruction = np.transpose(reconstruction[0,:,:,:], [1,2,0])
				reconstruction = crop_edges(reconstruction)
				etime = time.time()
				print("Done in {:.01f}s.".format(etime - stime))

			crop_name = os.path.join(directory, "{}-{}".format(basename, config.model.name))
													
			# Save results.
			saveExr(reconstruction, crop_name + '.exr')
			
			tasks.append(executor.submit(saveExr, reconstruction, crop_name + '.exr'))
			tasks.append(executor.submit(image.save_npz, reconstruction, crop_name + '.npz'))
			tasks.append(executor.submit(image.save_png, reconstruction, crop_name + '.png'))


def run_metrics(): # TODO HOX.
	import tensorflow as tf
	import elpips
	import darc
	import csv
	
	# Build graph.
	tf_X_input = tf.placeholder(tf.float32)
	tf_Y_input = tf.placeholder(tf.float32)
	tf_X = tf.expand_dims(tf_X_input, axis=0)
	tf_Y = tf.expand_dims(tf_Y_input, axis=0)
	
	tf_Y_grayscale = tf.reduce_mean(tf_Y, axis=3, keepdims=True)
	
	tf_l2 = tf.reduce_mean(tf.square(tf_X - tf_Y))
	tf_l1 = tf.reduce_mean(tf.abs(tf_X - tf_Y))
	tf_relmse = tf.reduce_mean(tf.square(tf_X - tf_Y) / (0.001 + tf.square(tf_Y_grayscale)))
	
	# Note: It would be somewhat faster to just use n=args.elpips_sample_count but TF has
	#       problems with n > 1 on some GPUs.
	elpips_vgg_model = elpips.Metric(elpips.elpips_vgg(n=1), back_prop=False)
	tf_elpips_vgg = elpips_vgg_model.forward(tf_X, tf_Y)[0]
	
	print("Creating Tensorflow session.")
		
	tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8

	with tf.Session(config=tf_config) as sess:
		# Initialize model.
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# Iterate over the archives.
		tasks = collections.deque()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:			
			for archive_path in DATASETS:
				# Reconstruct scene.
				scene_name = get_scene_name(archive_path)

				current_darc = darc.DataArchive(archive_path)
				
				crop_count = current_darc[0].shape[0] - 1
				image_count = len(current_darc)
								
				# Read minibatches.
				for image_index in range(image_count):
					# Execute previous tasks.
					while tasks:
						task = tasks[0].result()
						tasks.popleft()

					print("Loading reference... ", end="")
					sys.stdout.flush()
					stime = time.time()
					reference = current_darc[image_index][-1, :, :, 0:3]
					etime = time.time()
					print("Done in {:.01f}s.".format(etime - stime))
					
					directory = os.path.join(OUT_DIRECTORY, config.model.name, scene_name)
					
					if not os.path.exists(directory):
						print("Skipping directory '{}': Directory does not exist.".format(directory))
						continue
					
					for crop_index in range(crop_count):
						if args.crops and crop_index not in args.crops:
							print("(Skipping scene {}, image {}, crop {}).".format(scene_name, image_index, crop_index))
							continue

						crop_path = os.path.join(directory, "img{:04d}_crop{:02d}".format(image_index, crop_index))
							
						if not os.path.exists(crop_path + ".npz"):
							print("Skipping: Not found.")
							continue

						with open(os.path.join(directory, "img{:04d}_results.{}.csv".format(image_index, crop_index)), 'w') as csvfile:
							fields = ['crop_index', 'l1', 'l2', 'relmse', 'elpips-vgg', 'elpips-vgg-stdev']
							csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
							csv_writer.writeheader()
													
							print("Handling scene {}, image {}, crop {}.".format(scene_name, image_index, crop_index))
							
							# Load image.
							print("Loading image... ", end="")
							sys.stdout.flush()
							stime = time.time()

							current_image = image.load_npz(crop_path + ".npz")
							etime = time.time()
							print("Done in {:.01f}s.".format(etime - stime))
					
							# Run metrics.
							print("Running metrics... ", end="")
							sys.stdout.flush()
							stime = time.time()
							err_l1, err_l2, err_relmse = sess.run([tf_l1, tf_l2, tf_relmse], feed_dict={
								tf_X_input: current_image,
								tf_Y_input: reference
							})
								
							print_dot()
							
							err_elpips_vgg = []
							for i in range(args.elpips_sample_count):
								if i > 0 and i % 10 == 0:
									print_dot()
								
								err_elpips_vgg_single = sess.run(tf_elpips_vgg, feed_dict={
									tf_X_input: current_image,
									tf_Y_input: reference
								})
								err_elpips_vgg.append(err_elpips_vgg_single)
							
							err_elpips_vgg_mean = np.mean(err_elpips_vgg)
							err_elpips_vgg_std = np.std(err_elpips_vgg, ddof=1) / np.sqrt(args.elpips_sample_count)
							
							etime = time.time()
							print("Done in {:.01f}s.".format(etime - stime))
							
							# Save results.
							csv_writer.writerow({'crop_index': crop_index, 'l1': err_l1, 'l2': err_l2, 'relmse': err_relmse, 'elpips-vgg': err_elpips_vgg_mean, 'elpips-vgg-stdev': err_elpips_vgg_std})


			
def run():
	if args.reconstruct:
		reconstruct_one(args.reconstruct)
	if args.run_metrics:
		run_metrics() # TODO HOX.
	
						
if __name__ == '__main__':
	run()
	
