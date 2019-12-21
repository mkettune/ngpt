import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.client import device_lib

import datetime
import os
import sys
import pdb
import time
import random
import collections
import re

import config

from util import in_tuples, TimerCollection, benchmark_iteration, digit_reversed_range
import image
import archive
import model

import html_gen as html

import multiprocessing
import logging
mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)


if __name__ == '__main__':
	# Validate the model configuration.
	config.model.validate()

	# Initialize timers.
	timers = TimerCollection(skip_first=0)
	timers.setSkipFirst('export_html', 1)
	
	# Pre-create HTML/PNG workers for HTML reporting as early as possible to minimize multiprocessing's memory overhead.
	print("Creating PNG compressor workers.")
	gen = html.HtmlGenerator(config.HTML_DIR)
	gen.precreateCompressor(channels=3, width=config.CROP_SIZE, height=config.CROP_SIZE)
	print("Done with PNG compressors.")
		
	#print("**** BEGIN DEVICE LIST ****")
	#print(device_lib.list_local_devices())
	#print("**** END DEVICE LIST ****")

	print("**** Model index {} ****".format(config.MODEL_INDEX))
	
	# Read whether to restore weights.
	RESTORE_WEIGHTS = True if "--restore" in sys.argv[1:] else False
	if RESTORE_WEIGHTS:
		print("*** Restoring previous weights! ****")
	else:
		print("**** Not restoring weights. ****")

	# Create the database readers as early as possible.
	train_archive = archive.MultiFileArchive(config.TRAIN_DATASET_PATHS)
	
	print("Creating parallel epoch workers.")
	training_parallel_epoch = archive.preinit_parallel_epoch(train_archive, tuple_size=1, mode=config.PARALLEL_EPOCH_MODE)
	print("Created parallel epoch workers.")
	
	# Open the other databases.
	validation_archive = archive.MultiFileArchive(config.VALIDATION_DATASET_PATHS, validation=True)
	visualization_archive = archive.MultiFileArchive(config.VISUALIZATION_DARC_PATHS, validation=True)
	visualization_batch_divider = archive.BatchDivider(visualization_archive)
	
	# Get a deterministic visualization batch.
	print("Creating visualization batch.")
	random_state = random.getstate()
	random.seed(1)
	
	visualize_batch_count = (config.VISUALIZE_COUNT + config.BATCH_SIZE - 1) // config.BATCH_SIZE
	visualized_indices = digit_reversed_range(
		max(visualization_archive.image_count, config.VISUALIZE_COUNT),
		base=max(2, len(visualization_archive.archives))
	)[:config.VISUALIZE_COUNT]
	
	visualization_batches = []
	for batch_index in range(visualize_batch_count):
		indices = [visualized_indices[i] % visualization_archive.image_count for i in range(batch_index * config.BATCH_SIZE, min(config.VISUALIZE_COUNT, (batch_index+1) * config.BATCH_SIZE))]
			
		batch = visualization_batch_divider.constructBatch(indices)
		
		visualization_batches.append(batch)
			
	# Get deterministic validation batches.
	print("Reading validation batches to memory.")
	assert config.VALIDATION_SUMMARY_EXAMPLE_COUNT % config.BATCH_SIZE == 0
	
	validation_batch_divider = archive.DigitReversingBatchDivider(validation_archive)
	validation_batches = []
	index_begin = 0
	index_end = 0
	
	while index_end + config.BATCH_SIZE <= config.VALIDATION_SUMMARY_EXAMPLE_COUNT:
		index_end = index_begin + config.BATCH_SIZE
		print("{}..{}".format(index_begin, index_end))
		validation_batches.append(validation_batch_divider.constructBatch(validation_batch_divider.produceBatchIndices(config.BATCH_SIZE)))
		index_begin = index_end
	print("Done.")
		
	random.setstate(random_state)
	
	
	# Build network inputs.
	print("Creating GPU inputs.")
	with tf.device('/gpu:0'):
		with tf.name_scope('gpu0_input'):
			input_noisy = tf.Variable(tf.constant(0.0, shape=[config.BATCH_SIZE, config.INPUT_COUNT, config.CROP_SIZE, config.CROP_SIZE]), validate_shape=False, name='input_gpu0')
			
			input_clean = tf.Variable(tf.constant(0.0, shape=[config.BATCH_SIZE, config.TOTAL_COUNT - config.INPUT_COUNT, config.CROP_SIZE, config.CROP_SIZE]), validate_shape=False, name='input_clean_gpu0')
		
		gpu_inputs = {
			'input_noisy': input_noisy,
			'input_clean': input_clean,
			'assign_inference_input': model.assign_inference_input(input_noisy),
			'assign_training_input': model.assign_training_input(input_noisy, input_clean),
		}

	# Build the model.
	model_current = model.trainable_model(gpu_inputs, define_model_fn=model.reconstruct, compile_optimizer=True)
	
	
	# Build visualization tensors.
	print("Compiling util functions.")	
	
	def build_uint8_visualization_tensors():
		input_viz_ref = image.tf_to_nhwc(gpu_inputs['input_clean'][:,0:3,:,:])
		input_viz_ref_dx, input_viz_ref_dy = image.tf_evaluate_gradients(input_viz_ref)

		input_viz = image.tf_to_nhwc(gpu_inputs['input_noisy'][:,0:3,:,:])
		if config.model.input_enabled('gradients'):
			input_viz_dx, input_viz_dy = image.tf_to_nhwc(gpu_inputs['input_noisy'][:,3:6,:,:]), image.tf_to_nhwc(gpu_inputs['input_noisy'][:,6:9,:,:])
		else:
			input_viz_dx, input_viz_dy = image.tf_evaluate_gradients(input_viz_ref)
				
		return {
			'input_noisy': {
				'primal': image.tf_float_to_uint8(input_viz),
				'dx': image.tf_float_to_uint8(input_viz_dx + 0.5),
				'dy': image.tf_float_to_uint8(input_viz_dy + 0.5),
			},
			'input_clean': {
				'primal': image.tf_float_to_uint8(input_viz_ref),
				'dx': image.tf_float_to_uint8(input_viz_ref_dx + 0.5),
				'dy': image.tf_float_to_uint8(input_viz_ref_dy + 0.5),
			}
		}
		
	get_as_uint8 = build_uint8_visualization_tensors()

	
	# Output model details.
	print("Model '{}' has {} parameters.".format(model_current['name'], model_current['parameter_count']))
	
		
	# Run training.
	tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

	#pdb.set_trace()	
	print("Creating Tensorflow session.")	
	
	with tf.Session(config=tf_config) as sess:
		# Initialize the model.
		print("Initializing global variables.")
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		
		# Restore weights if asked.
		if RESTORE_WEIGHTS:
			print("**** Restoring model weights! ****")
			
			print("Loading model {}... ".format(model_current['name']), end="")
			sys.stdout.flush()
			model_current['saver'].restore(sess, os.path.join(config.SAVE_DIRECTORY, "{}.ckpt".format(model_current['name'])))
			print("Done.")
		
			# Load session name.
			session_name = sess.run(model_current['session_name']).decode("utf-8")
		
		else:
			# Generate session name.
			session_name = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
			sess.run([tf.assign(model_current['session_name'], session_name)])
	
		# Initialize statistics writing.
		os.makedirs(config.SUMMARY_DIR, exist_ok=True)
		model_current['train_writer'] = tf.summary.FileWriter(os.path.join(config.SUMMARY_DIR, "%s %s train" % (session_name, model_current['name'])), graph=None)
		model_current['validation_writer'] = tf.summary.FileWriter(os.path.join(config.SUMMARY_DIR, "%s %s validation" % (session_name, model_current['name'])))
		
		# Pre-convert visualization minibatches to uint8 images.
		if True:
			print("Reading visualization results from GPU.")
			
			visualization_input_noisy_primal = []
			visualization_input_noisy_dx = []
			visualization_input_noisy_dy = []
			visualization_input_clean_primal = []
			visualization_input_clean_dx = []
			visualization_input_clean_dy = []

			for visualization_batch in visualization_batches:
				# Set input minibatch.
				sess.run(
					gpu_inputs['assign_training_input']['kernel'],
					feed_dict={gpu_inputs['assign_training_input']['data']: visualization_batch}
				)
				
				# Convert to uint8.
				input_noisy_primal, input_noisy_dx, input_noisy_dy, input_clean_primal, input_clean_dx, input_clean_dy = sess.run([
					get_as_uint8['input_noisy']['primal'], get_as_uint8['input_noisy']['dx'], get_as_uint8['input_noisy']['dy'],
					get_as_uint8['input_clean']['primal'], get_as_uint8['input_clean']['dx'], get_as_uint8['input_clean']['dy']
				])
				
				def removePad(x):
					return x[:, config.PAD_WIDTH : x.shape[1] - config.PAD_WIDTH, config.PAD_WIDTH : x.shape[2] - config.PAD_WIDTH, :]
					
				visualization_input_noisy_primal.append(removePad(input_noisy_primal))
				visualization_input_noisy_dx.append(removePad(input_noisy_dx))
				visualization_input_noisy_dy.append(removePad(input_noisy_dy))
				visualization_input_clean_primal.append(removePad(input_clean_primal))
				visualization_input_clean_dx.append(removePad(input_clean_dx))
				visualization_input_clean_dy.append(removePad(input_clean_dy))
				
			visualization_input_noisy_primal = np.concatenate(visualization_input_noisy_primal, axis=0)
			visualization_input_noisy_dx = np.concatenate(visualization_input_noisy_dx, axis=0)
			visualization_input_noisy_dy = np.concatenate(visualization_input_noisy_dy, axis=0)
			visualization_input_clean_primal = np.concatenate(visualization_input_clean_primal, axis=0)
			visualization_input_clean_dx = np.concatenate(visualization_input_clean_dx, axis=0)
			visualization_input_clean_dy = np.concatenate(visualization_input_clean_dy, axis=0)
			
			
		# Run training.		
		batch_index = sess.run(model_current['batch_index'])
		visualization_counter = -float(batch_index) + config.VISUALIZATION_PERIOD
		visualization_period = config.VISUALIZATION_PERIOD

		
		timers['total'].begin()
		
		
		print("Starting training minibatch workers.")		
		for training_minibatches in benchmark_iteration(training_parallel_epoch(minibatches=config.LAST_MINIBATCH-batch_index), timer=timers['iterator_train']):
			last_minibatch = (config.LAST_MINIBATCH >= 0 and batch_index + 1 == config.LAST_MINIBATCH)
						
			# Send training data to the GPUs.
			timers['send_training_data'].begin()
			
			assign_op = gpu_inputs['assign_training_input']
			sess.run(assign_op['kernel'], {assign_op['data']:training_minibatches[0]})
			
			timers['send_training_data'].end()
			
			# Run training.
			run_training_summary = last_minibatch or (batch_index % config.TRAINING_SUMMARY_PERIOD) < 1


			if True:
				### Update learning rate.
				
				timers['update learning rate'].begin()
				sess.run(model_current['update_learning_rate'])
				timers['update learning rate'].end()
				

				### Run training.				
				
				timers['collect training kernels'].begin()
				
				kernels = []
				ops = []
				feed_dict = {}
				
				summaries = {}
				metrics = {}
				
				# Add training ops to be run.
				for train_op_name, train_op in model_current['train_ops'].items():
					kernels.append(train_op)
					ops.append(lambda x: 0.0)

				# Add training summaries to be run.
				if run_training_summary:
					print("Running (also) training summary for '{}'.".format(model_current['name']))
				
					if 'summaries' in model_current:
						kernels.append(model_current['summaries'])
						def op(x, name=model_current['name']):
							summaries[name] = x
						ops.append(op)
						
					kernels.append(model_current['metrics'])
					def op(x, name=model_current['name']):
						metrics[name] = x
					ops.append(op)
				
				timers['collect training kernels'].end()
			
				# Run the training kernels.
				timers['run training kernels'].begin()
				results = sess.run(kernels, feed_dict=feed_dict)
				timers['run training kernels'].end()
				
				timers['run training ops'].begin()
				for x, op in zip(results, ops):
					op(x)					
				timers['run training ops'].end()

				# Add summaries to TensorBoard.
				if run_training_summary:
					timers['handle training summary'].begin()
										
					if model_current['name'] in summaries:
						training_summary = summaries[model_current['name']]
						model_current['train_writer'].add_summary(training_summary, batch_index)
						
					model_metrics = metrics[model_current['name']]
					
					summary = tf.Summary()
					
					for key, value in model_metrics.items():
						summary.value.add(tag='metrics/{}'.format(key), simple_value=value)
					
					summary.value.add(tag='metrics/minibatch_index', simple_value=batch_index)

					timer = timers.getTimer('run training kernels')
					if timer:
						summary.value.add(tag='metrics/time_training', simple_value=timer.total())

					timer = timers.getTimer('total')
					if timer:
						summary.value.add(tag='metrics/time_total', simple_value=timer.total())
						
					timer = timers.getTimer('send_training_data')
					if timer:
						summary.value.add(tag='metrics/time_data_upload', simple_value=timer.total())
					
					timer = timers.getTimer('iterator_train')
					if timer:
						summary.value.add(tag='metrics/time_data_wait', simple_value=timer.total())
						
					model_current['train_writer'].add_summary(summary, batch_index)							
					model_current['train_writer'].flush()
					
					timers['handle training summary'].end()
					

			### Run validation.
			
			run_validation_summary = last_minibatch or (batch_index % config.VALIDATION_SUMMARY_PERIOD) < 1

			if run_validation_summary:
				timers['validation summary'].begin()
				
				print("Running {} validation minibatches...".format(config.VALIDATION_SUMMARY_EXAMPLE_COUNT), end="")
				sys.stdout.flush()
            	
				# Collect the validation operations.
				metrics_sums = collections.defaultdict(float)
				metrics_weights = collections.defaultdict(float)
				summaries = {}
				validation_kernels = []
				validation_ops = []
				validation_feed_dict = {}

				# Add collection of summaries.
				if 'summaries' in model_current:
					validation_kernels.append(model_current['summaries'])
					def op(x, name=model_current['name']):
						summaries[name] = x
					validation_ops.append(op)
				
				# Add collection of loss metrics.
				validation_kernels.append(model_current['metrics'])
				def op(x, name=model_current['name']):
					for key, value in x.items():
						metrics_sums[key] += value
						metrics_weights[key] += 1.0
				validation_ops.append(op)

				# Run the validation ops for all validation minibatches.
				print(" ", end="")
				for validation_index, minibatch in enumerate(validation_batches):				
					# Send the minibatch to the GPU.
					sess.run(
						gpu_inputs['assign_inference_input']['kernel'],
						feed_dict={gpu_inputs['assign_inference_input']['data']: minibatch}
					)
					
					# Run validation for the minibatch.
					for x, op in zip(sess.run(validation_kernels, feed_dict=validation_feed_dict), validation_ops):
						op(x)
						
					print(".", end="")

				# Output metrics to TensorBoard.			
				if model_current['name'] in summaries:
					validation_summary = summaries[model_current['name']]
					model_current['validation_writer'].add_summary(validation_summary, batch_index)
				
				model_metrics = {}
				for key in metrics_sums.keys():
					model_metrics[key] = metrics_sums[key] / metrics_weights[key]
					
				summary = tf.Summary()
					
				for key, value in model_metrics.items():
					summary.value.add(tag='metrics/{}'.format(key), simple_value=value)
				
				model_current['validation_writer'].add_summary(summary, batch_index)
				model_current['validation_writer'].flush()
				
				timers['validation summary'].end()
						
				print(" Done.")

				
			### Minibatch finished.
			
			if batch_index % 10 == 0:
				print("Finished minibatch %d of '%s %s'." % (batch_index, session_name, model_current['name']))
			

			### Output HTML visualization.
			
			visualization_counter -= 1
			if last_minibatch or visualization_counter < 0.5:
				timers['export_html'].begin()
				print("Exporting HTML...", end="")
				sys.stdout.flush()
            
				# Update the time of next visualization.					
				while visualization_counter < 0.5:
					visualization_counter += float(visualization_period)
					visualization_period *= config.VISUALIZATION_PERIOD_MULTIPLIER
			
				# Initialize the generated page.
				gen.setTitle(title="Results for {} {}".format(session_name, model_current['name']))

				info = gen.info('Result Info')
				info.setMessage('Batch {}'.format(batch_index + 1))

				html_visualization = gen.imageComparison(name='Validation', default_group='input')
				
				# Add the input images.
				html_visualization.add('input',
					(visualization_input_noisy_primal, visualization_input_noisy_dx, visualization_input_noisy_dy),
					static=True
				)
				
				# Add the reconstructions.
				result_primal = []
				result_dx = []
				result_dy = []
		
				for visualization_batch_input in visualization_batches:
					# Set the minibatch.
					sess.run(
						gpu_inputs['assign_inference_input']['kernel'],
						feed_dict={gpu_inputs['assign_inference_input']['data']: visualization_batch_input}
					)

					# Get the reconstructions.
					primal, dx, dy = sess.run([
						model_current[0]['y_uint8'],
						model_current[0]['dx_uint8'],
						model_current[0]['dy_uint8'],
					])
					
					result_primal.extend(primal)
					result_dx.extend(dx)
					result_dy.extend(dy)
				
				result_primal = np.stack(result_primal, axis=0)
				result_dx = np.stack(result_dx, axis=0)
				result_dy = np.stack(result_dy, axis=0)
									
				html_visualization.add(
					model_current['name'],
					(removePad(result_primal), removePad(result_dx), removePad(result_dy))
				)
				
				# Add the reference.				
				html_visualization.add(
					'reference',
					(visualization_input_clean_primal, visualization_input_clean_dx, visualization_input_clean_dy),
					static=True
				)
            
				gen.write()
				
				print(" Done.")
				timers['export_html'].end()

			# Save.
			if (batch_index > 0 and (batch_index % config.SAVING_PERIOD) < 1) or last_minibatch:
				timers['save'].begin()
				
				print("Saving model {}... ".format(model_current['name']), end="")
				sys.stdout.flush()

				os.makedirs(config.SAVE_DIRECTORY, exist_ok=True)
				model_current['saver'].save(sess, os.path.join(config.SAVE_DIRECTORY, "{}.ckpt".format(model_current['name'])), write_meta_graph=False)
					
				print("Done.")
				
				timers['save'].end()
			
			batch_index += 1
			timers['total'].end()
			
			# Report timings.
			if batch_index % 10 == 1:
				timers.report()
			
			if last_minibatch:
				model_current['train_writer'].flush()
				model_current['validation_writer'].flush()
			
			timers['total'].begin()
			
	del gen
	gen = None
	
