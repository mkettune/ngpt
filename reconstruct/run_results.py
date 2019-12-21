import numpy as np

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

from util import *
import image
import archive
import model

import imageio


### Output directory.

OUT_DIRECTORY = os.path.join(config.BASE_DIR, 'results', 'base_model')


### Datasets to reconstruct.

if config.model.input_enabled('gradients'):
	# Full size image datasets for gradient-domain reconstruction mode.
	DATASETS = [
#		config.DATASET_DIR + "/test_gpt-bookshelf_rough2-2-12.darc",
		config.DATASET_DIR + "/test_gpt-new_bedroom-2-12.darc",
		config.DATASET_DIR + "/test_gpt-new_dining_room-2-12.darc",
		config.DATASET_DIR + "/test_gpt-new_kitchen_animation-2-12.darc",
		config.DATASET_DIR + "/test_gpt-new_kitchen_dof-2-12.darc",
		config.DATASET_DIR + "/test_gpt-running_man-2-12.darc",
		config.DATASET_DIR + "/test_gpt-sponza-2-12.darc",
	]
else:
	# Full size image datasets for primal-domain reconstruction mode.
	DATASETS = [
#		config.DATASET_DIR + "/test_pt-bookshelf_rough2-2-12.darc",
		config.DATASET_DIR + "/test_pt-new_bedroom-2-12.darc",
		config.DATASET_DIR + "/test_pt-new_dining_room-2-12.darc",
		config.DATASET_DIR + "/test_pt-new_kitchen_animation-2-12.darc",
		config.DATASET_DIR + "/test_pt-new_kitchen_dof-2-12.darc",
		config.DATASET_DIR + "/test_pt-running_man-2-12.darc",
		config.DATASET_DIR + "/test_pt-sponza-2-12.darc",
	]

	
def get_scene_name(archive_path):
	return os.path.splitext(os.path.basename(archive_path))[0].replace("-pt-", "-")

def print_dot():
	print(".", end="")
	sys.stdout.flush()

	
### Command line options.
	
parser = argparse.ArgumentParser()
parser.add_argument('--crops', type=int, nargs='+', help='which crop indices to evaluate (default: 0 1 2 ... 11)')
parser.add_argument('--config', type=int, help='config index to reconstruct or whose dataset to extract.')
parser.add_argument('--reconstruct', action='store_true', help='reconstruct the given crops of all datasets for the given model.')
parser.add_argument('--extract-dataset', action='store_true', help='extract the inputs and ground truths of the dataset used by the given model.')
parser.add_argument('--run-metrics', action='store_true', help='run measurement metrics for the given crops of all datasets of the given model.')
parser.add_argument('--elpips-sample-count', type=int, default=200, help='number of samples for E-LPIPS measurement. Increase to lower stdev. Default: 200. Should probably be at least 50.')
args = parser.parse_args()
assert args.elpips_sample_count >= 50


# Validate the model configuration.
config.model.validate()


def reconstruct_all():
	'''Reconstructs all images in the input dataset.'''
	import tensorflow as tf

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
	model_current = model.trainable_model(gpu_inputs, define_model_fn=model.reconstruct, compile_optimizer=False)
	

	# Setting minibatch.
	assign_minibatch = model.assign_inference_input(gpu_inputs['input_noisy'])

	# Start TF session.
	tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	print("Creating Tensorflow session.")	

	with tf.Session(config=tf_config) as sess:
		# Initialize the model.
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		print("**** Restoring model weights! ****")
		print("Loading model {}... ".format(model_current['name']), end="")
		sys.stdout.flush()
		model_current['saver'].restore(sess, os.path.join(config.SAVE_DIRECTORY, "{}.ckpt".format(model_current['name'])))
		print("Done.")
	
		# Run results for the dataset.
		tasks = collections.deque()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:			
			for archive_path in DATASETS:
				# Execute the previous parallel tasks.
				for task in tasks:
					task.result()
				tasks = collections.deque()

				# Reconstruct from the inputs.
				scene_name = get_scene_name(archive_path)
			
				current_archive = archive.Archive(archive_path, validation=True)
				crop_count = current_archive.crop_count
				image_count = current_archive.image_count // crop_count
				
				# Read images from the dataset.
				for image_index in range(image_count):
					for crop_index in range(crop_count):
						if not (not args.crops or crop_index in args.crops):
							#print("Skipping scene {}, image {}, crop {}.".format(scene_name, image_index, crop_index))
							continue
						
						print("Handling scene {}, image {}, crop {}.".format(scene_name, image_index, crop_index))
						
						print("Loading data... ", end="")
						stime = time.time()
						sys.stdout.flush()
						
						# TODO: Parallelize dataset reading for performance.
						minibatch = current_archive.readCrop(image_index * crop_count + crop_index)

						minibatch = np.expand_dims(minibatch[:-3], 0)
						etime = time.time()
						print("Done in {:.01f}s.".format(etime - stime))
						
						# Send training data.
						print("Reconstructing... ", end="")
						sys.stdout.flush()
						stime = time.time()
						sess.run(assign_minibatch['kernel'], feed_dict={assign_minibatch['data']: minibatch})
						
						# Run reconstruction.
						reconstruction = sess.run(model_current[0]['y'])
						reconstruction = np.transpose(reconstruction[0,:,:,:], [1,2,0])
						reconstruction = crop_edges(reconstruction)
						etime = time.time()
						print("Done in {:.01f}s.".format(etime - stime))

						directory = os.path.join(OUT_DIRECTORY, model_current['name'], scene_name)
						crop_name = os.path.join(directory, "img{:04d}_crop{:02d}".format(image_index, crop_index))
													
						# Save results.
						os.makedirs(directory, exist_ok=True)
						tasks.append(executor.submit(image.save_npz, reconstruction, crop_name + '.npz'))
						tasks.append(executor.submit(image.save_png, reconstruction, crop_name + '.png'))


def extract_dataset():
	import darc
		
	if config.model.input_enabled('gradients'):
		out_dir_input = "input-gpt"
		out_dir_reference = "references"
	else:
		out_dir_input = "input-pt-2.5x"
		out_dir_reference = "references"
		
	# Iterate over the archives.
	tasks = collections.deque()
	with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:			
		for archive_path in DATASETS:
			# Get scene inputs.
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

				print("Loading metadata... ", end="")
				stime = time.time()
				sys.stdout.flush()
				minibatch = current_darc[image_index]
				etime = time.time()
				print("Done in {:.01f}s.".format(etime - stime))
				
				# Extract inputs.
				directory = os.path.join(OUT_DIRECTORY, out_dir_input, scene_name)
				os.makedirs(directory, exist_ok=True)

				for crop_index in range(crop_count):
					print("Handling scene {}, image {}, crop {}.".format(scene_name, image_index, crop_index))

					print("Extracting inputs... ", end="")
					stime = time.time()
					
					def getSlice(which):
						return minibatch[crop_index, :, :, which[0]:which[1]]
										
					def saveSlice(data, slice_name):
						tasks.append(executor.submit(image.save_png, data, os.path.join(directory, "img{:04d}_crop{:02d}_{}.png".format(image_index, crop_index, slice_name))))
						tasks.append(executor.submit(image.save_npz, data, os.path.join(directory, "img{:04d}_crop{:02d}_{}.npz".format(image_index, crop_index, slice_name))))
		
					saveSlice(getSlice(config.IND_PRIMAL_COLOR), 'primal')
					print_dot()
					saveSlice(getSlice(config.IND_DX_COLOR), 'dx')
					print_dot()
					saveSlice(getSlice(config.IND_DY_COLOR), 'dy')
					print_dot()
					saveSlice(getSlice(config.IND_ALBEDO), 'albedo')
					print_dot()
					saveSlice(getSlice(config.IND_DEPTH), 'depth')
					print_dot()
					saveSlice(getSlice(config.IND_NORMAL), 'normal')
					print_dot()
					saveSlice(getSlice(config.IND_DIFFUSE), 'diffuse')
					print_dot()
					saveSlice(getSlice(config.IND_SPECULAR), 'specular')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_PRIMAL), 'var_primal')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_DX), 'var_dx')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_DY), 'var_dy')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_ALBEDO), 'var_albedo')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_DEPTH), 'var_depth')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_NORMAL), 'var_normal')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_DIFFUSE), 'var_diffuse')
					print_dot()
					saveSlice(getSlice(config.IND_VAR_SPECULAR), 'var_specular')
					
					etime = time.time()
					print("Done in {:.01f}s.".format(etime - stime))

				# Extract references.
				print("Extracting reference... ", end="")
				stime = time.time()

				directory = os.path.join(OUT_DIRECTORY, out_dir_reference, scene_name)
				os.makedirs(directory, exist_ok=True)

				crop_name = os.path.join(directory, "img{:04d}_ref".format(image_index))
				tasks.append(executor.submit(image.save_npz, minibatch[crop_count, :, :, 0:3], crop_name + '.npz'))
				tasks.append(executor.submit(image.save_png, minibatch[crop_count, :, :, 0:3], crop_name + ".png"))
				
				etime = time.time()
				print("Done in {:.01f}s.".format(etime - stime))


def run_metrics():
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
	tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7

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
		reconstruct_all()
	if args.extract_dataset:
		extract_dataset()
	if args.run_metrics:
		run_metrics()
		
	if not args.reconstruct and not args.extract_dataset and not args.run_metrics:
		print("No action given. See --help.")
	
						
if __name__ == '__main__':
	run()
	
