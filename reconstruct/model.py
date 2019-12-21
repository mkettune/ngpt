import tensorflow as tf
import numpy as np

import archive
import layers
import config
import image
import itertools
import pdb
import sys
import time

from util import scope

import elpips


def run_metrics(prediction, target_x, target_dx=None, target_dy=None, source=None):
	with tf.name_scope('metrics'):
	
		def to_ldr_nhwc(x):
			'''Prepares an image for the perceptual losses.'''
			x = tf.maximum(0.0, x)
			x = layers.srgb_to_nonlinear(x)			
			x = image.tf_to_nhwc(x)
			return x
		
		elpips_vgg_config = elpips.elpips_vgg(config.BATCH_SIZE)
		elpips_vgg_config.fast_and_approximate = True
		elpips_vgg_config.set_scale_levels(2)

		elpips_squeezenet_config = elpips.elpips_squeeze_maxpool(config.BATCH_SIZE)
		elpips_squeezenet_config.fast_and_approximate = True
		elpips_squeezenet_config.set_scale_levels(2)
		
		if config.model.elpips_eval_count is not None:
			elpips_vgg_config.average_over = config.model.elpips_eval_count
			elpips_squeezenet_config.average_over = config.model.elpips_eval_count

		elpips_vgg = elpips.Metric(elpips_vgg_config)
		elpips_squeeze_maxpool = elpips.Metric(elpips_squeezenet_config)
		
		lpips_vgg = elpips.Metric(elpips.lpips_vgg(config.BATCH_SIZE))
		lpips_squeeze = elpips.Metric(elpips.lpips_squeeze(config.BATCH_SIZE))

			
		assert config.PAD_WIDTH > 0
		if config.PAD_WIDTH > 0:
			shape = tf.shape(prediction)
			N,C,H,W = shape[0], shape[1], shape[2], shape[3]
			
			X0, Y0 = config.PAD_WIDTH + config.model.vary_padding, config.PAD_WIDTH + config.model.vary_padding
			X1, Y1 = W - config.PAD_WIDTH - config.model.vary_padding, H - config.PAD_WIDTH - config.model.vary_padding
			
			prediction = prediction[:, :, Y0:Y1, X0:X1]
			target_x = target_x[:, :, Y0:Y1, X0:X1]
			
			if target_dx is not None:
				target_dx = target_dx[:, :, Y0:Y1, X0:X1]
				
			if target_dy is not None:
				target_dy = target_dy[:, :, Y0:Y1, X0:X1]
				
			if source is not None:
				source = source[:, :, Y0:Y1, X0:X1]
					
		l1_error = tf.losses.absolute_difference(target_x, prediction)
		
		prediction_reinhard = prediction / (1.0 + tf.reduce_mean(tf.abs(prediction), axis=1, keepdims=True))
		target_reinhard = target_x / (1.0 + tf.reduce_mean(tf.abs(target_x), axis=1, keepdims=True))
		
		l1_tonemap_error = tf.losses.absolute_difference(target_reinhard, prediction_reinhard)
		
		mean_color_prediction = tf.reduce_mean(prediction, axis=[2, 3])
		mean_color_target_x = tf.reduce_mean(target_x, axis=[2, 3])
		mean_color_error_l1 = tf.reduce_mean(tf.abs(mean_color_prediction - mean_color_target_x))
		
		negative_loss = tf.reduce_mean(tf.maximum(-prediction, 0.0))
		
		# RelMSE.
		def RelMSE(prediction, reference):
			EPSILON = 0.001
			grayscale_reference = tf.reduce_mean(reference, axis=1, keepdims=True)
			image_error = prediction - reference
			
			relmse_image = tf.reduce_mean(tf.square(image_error), axis=1, keepdims=True) / (EPSILON + tf.square(grayscale_reference))
			return tf.reduce_mean(relmse_image, axis=[0, 1, 2, 3])

		relmse = RelMSE(prediction, target_x)
	
		# Perceptual-tonemap-sRGB
		perceptual_prediction = to_ldr_nhwc(prediction_reinhard)
		perceptual_target = to_ldr_nhwc(target_reinhard)

		elpips_squeeze_maxpool_loss = tf.reduce_mean(elpips_squeeze_maxpool.forward(perceptual_prediction, perceptual_target))
		elpips_vgg_loss = tf.reduce_mean(elpips_vgg.forward(perceptual_prediction, perceptual_target))
		lpips_squeeze_loss = tf.reduce_mean(lpips_squeeze.forward(perceptual_prediction, perceptual_target))
		lpips_vgg_loss = tf.reduce_mean(lpips_vgg.forward(perceptual_prediction, perceptual_target))
		
		metrics = {
			'L1': l1_error,
			'L1_tonemap': l1_tonemap_error,
			'RelMSE': relmse,
			'elpips_squeeze_maxpool': elpips_squeeze_maxpool_loss,
			'elpips_vgg': elpips_vgg_loss,
			'lpips_squeeze': lpips_squeeze_loss,
			'lpips_vgg': lpips_vgg_loss,
			'mean_color_L1': mean_color_error_l1,
			'negative_loss': negative_loss
		}
						
		if target_dx is not None and target_dy is not None:
			prediction_dx = layers.dx(prediction)
			prediction_dy = layers.dy(prediction)

			prediction_dx_reinhard = prediction_dx / (1.0 + tf.reduce_mean(tf.abs(prediction_dx), axis=1, keepdims=True))
			prediction_dy_reinhard = prediction_dy / (1.0 + tf.reduce_mean(tf.abs(prediction_dy), axis=1, keepdims=True))
			target_dx_reinhard = target_dx / (1.0 + tf.reduce_mean(tf.abs(target_dx), axis=1, keepdims=True))
			target_dy_reinhard = target_dy / (1.0 + tf.reduce_mean(tf.abs(target_dy), axis=1, keepdims=True))
			
			gradient_l1_error = (tf.losses.absolute_difference(target_dx, prediction_dx) + tf.losses.absolute_difference(target_dy, prediction_dy))
			metrics['grad_L1'] = gradient_l1_error
		
			gradient_l1t_error = (tf.losses.absolute_difference(target_dx_reinhard, prediction_dx_reinhard) + tf.losses.absolute_difference(target_dy_reinhard, prediction_dy_reinhard))

			metrics['grad_L1_tonemap'] = gradient_l1t_error
			
	return metrics


def get_pool():
	pool_type = config.model.pool_type

	if pool_type == 'conv':
		pool = lambda p1, p2: layers.pool_conv(p1, p2, relu_factor=config.model.relu_factor)
	elif pool_type == 'lanczos':
		pool = lambda p1, p2: layers.pool_lanczos(p1)
	elif pool_type == 'average':
		pool = lambda p1, p2: layers.pool_average(p1)
	elif pool_type == 'max':
		pool = lambda p1, p2: layers.pool_max(p1)
	else:
		raise Exception('unknown pool_type')
		
	return pool
	
def get_unpool():
	unpool_type = config.model.unpool_type
	
	if unpool_type == 'conv':
		unpool = lambda p1, p2: layers.unpool_conv(p1, p2, relu_factor=config.model.relu_factor)
	elif unpool_type == 'lanczos':
		unpool = lambda p1, p2: layers.unpool_lanczos(p1)
	elif unpool_type == 'nearest':
		unpool = lambda p1, p2: layers.unpool_nearest(p1)
	elif unpool_type == 'bilinear':
		unpool = lambda p1, p2: layers.unpool_bilinear(p1)
	else: 
		raise Exception('unknown unpool_type')
	
	return unpool
	
def get_conv():
	conv2d = lambda p1,p2,p3: layers.conv2d(p1, p2, p3, relu_factor=config.model.relu_factor)
	return conv2d
	
def get_conv_denselist():
	# Denselist: The tensor is given as a list [T1, ..., Tn], thought to be concatenated in the feature dimension.
	#            The concatenation is not applied since there is no actual need to construct the new tensor.
	dense_conv2d = lambda p1, p2, p3: layers.denselist_conv2d(p1, p2, p3, relu_factor=config.model.relu_factor)
	return dense_conv2d
	

def define_network(first_layer_tensor):
	'''This function defines the actual network.'''
	
	# Shortcuts to the model config.
	levels = config.model.levels
	k0, k1, k2, k3, k4 = config.model.growth_rates
	unit_count0, unit_count1, unit_count2, unit_count3, unit_count4 = config.model.unit_counts
	relu_factor = config.model.relu_factor
	downscale_feature_counts = config.model.downscale_feature_counts
	upscale_feature_counts = config.model.upscale_feature_counts

	# Get operations by the model config.
	conv2d = get_conv()
	dense_conv2d = get_conv_denselist()
	pool = get_pool()
	unpool = get_unpool()
	
	
	# Model building blocks.
	
	def unit(denselist, growth_rate, depth):
		# The 'Processing Unit'.
		with scope('Bottleneck'):
			current = dense_conv2d(denselist, 2 * growth_rate, 1)

		with scope('Conv0'):
			current = conv2d(current, growth_rate, 3)	
		
		return current
	
	def nUnits(input, growth_rate, unit_count, depth=1):
		# A sequence of 'Processing Units'.
		
		assert type(input) is list
		denselist = []
		denselist.extend(input)
		
		for i in range(unit_count):
			with scope('Unit{}'.format(i)):
				denselist.append(unit(denselist, growth_rate, depth))
				
		return denselist
	
	def densePool(denselist, depth):
		# Pooling to half-size and a different feature count.
		
		feature_count = downscale_feature_counts[depth]
		
		if config.model.pool_type == 'conv':
			# Convolution unpool directly to the correct feature count.
			current = layers.denselist_pool_conv(denselist, feature_count, relu_factor=relu_factor)
		else:
			# Convolve to the correct feature count, then pool.
			current = layers.denselist_conv2d(denselist, feature_count, 1)
			current = pool(current, None)
		
		return [current]
	
	def denseUnpool(denselist, previous, depth):
		# Unpooling to double size and a different feature count.
		
		assert type(denselist) is list
		assert type(previous) is list
		dense_conv2d = lambda p1, p2, p3: layers.denselist_conv2d(p1, p2, p3, relu_factor=relu_factor)
		
		feature_count = upscale_feature_counts[depth]

		if config.model.unpool_type == 'conv':
			# Convolution unpool directly to the correct feature count.
						
			input = layers.concat(denselist) # TODO: Use layers.denselist_unpool_conv to save memory.
			current = unpool(input, feature_count)
		else:
			# Convolve to the correct feature count, then unpool.
			current = layers.denselist_conv2d(denselist, feature_count, 1)
			current = unpool(current, None)
					
		result = []
		result.extend(previous)
		result.append(current)
		
		return result
	
	
	# Network definition.
	with scope('E0'): current0 = nUnits([first_layer_tensor], k0, unit_count0, 0)
	
	if levels > 1:
		with scope('E0_pool'): current1 = densePool(current0, 1)
		with scope('E1'): current1 = nUnits(current1, k1, unit_count1, 1)
		
		if levels > 2:
			with scope('E1_pool'): current2 = densePool(current1, 2)
			with scope('E2'): current2 = nUnits(current2, k2, unit_count2, 2)

			if levels > 3:
				with scope('E2_pool'): current3 = densePool(current2, 3)
				with scope('E3'): current3 = nUnits(current3, k3, unit_count3, 3)

				if levels > 4:
					assert levels <= 5
					with scope('E3_pool'): current4 = densePool(current3, 4)
					with scope('C0'): current4 = nUnits(current4, 2 * k4, unit_count4, 4)
					with scope('D3_unpool'): current3 = denseUnpool(current4, current3, 3)					
					
				with scope('D3'): current3 = nUnits(current3, k3, unit_count3, 3)
				with scope('D2_unpool'): current2 = denseUnpool(current3, current2, 2)
				
			with scope('D2'): current2 = nUnits(current2, k2, unit_count2,  2)
			with scope('D1_unpool'): current1 = denseUnpool(current2, current1, 1)
			
		with scope('D1'): current1 = nUnits(current1, k1, unit_count1, 1)
		with scope('D0_unpool'): current0 = denseUnpool(current1, current0, 0)
		
	with scope('D0'): current0 = nUnits(current0, k0, unit_count0, 0)

	with scope('R0_RGB'): current = layers.denselist_conv2d(current0, 3, 1, relu_factor=relu_factor)
	
		
	y = current
	
	# Undo log(1+x) of inputs.
	y = tf.exp(y) - 1.0
	
	return y
	
	
def construct_first_layer(gpu_inputs):
	# Collect indices (in the dataset) of the features that are enabled.
	first_layer = [config.IND_PRIMAL_COLOR] # primal
	
	if config.model.input_enabled('gradients'):
		first_layer.append(config.IND_DX_COLOR) # dx
		first_layer.append(config.IND_DY_COLOR) # dy
		
	if config.model.input_enabled('variances'):
		first_layer.append(config.IND_VAR_PRIMAL) # var_primal
		if config.model.input_enabled('gradients'):
			first_layer.append(config.IND_VAR_DX) # var_dx
			first_layer.append(config.IND_VAR_DY) # var_dx

	if config.model.input_enabled('albedo'):
		first_layer.append(config.IND_ALBEDO) # albedo
	
	if config.model.input_enabled('normal'):
		first_layer.append(config.IND_NORMAL) # normal
	
	if config.model.input_enabled('depth'):
		first_layer.append(config.IND_DEPTH) # depth
	
	if config.model.input_enabled('variances'):
		if config.model.input_enabled('albedo'):
			first_layer.append(config.IND_VAR_ALBEDO) # albedo
		
		if config.model.input_enabled('normal'):
			first_layer.append(config.IND_VAR_NORMAL) # normal
		
		if config.model.input_enabled('depth'):
			first_layer.append(config.IND_VAR_DEPTH) # depth

	# Construct the actual first layer tensor.
	input_feature_count = 0
	first_layer_features = []

	inputs = gpu_inputs['input_noisy']

	for feature_range in first_layer:
		input_feature_count += feature_range[1] - feature_range[0]
		mapped_range = config.get_minibatch_dims(feature_range)
		first_layer_features.append(inputs[:, mapped_range[0]:mapped_range[1], :, :])
	
	first_layer_tensor = tf.concat(first_layer_features, axis=1)
	first_layer_tensor.set_shape([None, input_feature_count, None, None])

	return first_layer_tensor
	

def reconstruct(gpu_inputs, evaluate_metrics=True):
	input = gpu_inputs['input_noisy']
	train_target = gpu_inputs['input_clean']
	
	# Note: The variable_scope name must match the name of the loss function in 'losses' below.
	with tf.variable_scope('reconstruct', reuse=tf.AUTO_REUSE): 
		target_x = train_target[:, config.IND_PRIMAL_COLOR[0]:config.IND_PRIMAL_COLOR[1], :, :]
		target_x.set_shape([None, 3, None, None])

		# Create the first layer.
		first_layer_tensor = construct_first_layer(gpu_inputs)
		
		# Run the network.
		y = define_network(first_layer_tensor=first_layer_tensor)
		
		# Evaluate metrics and loss.
		target_dx = layers.dx(target_x)
		target_dy = layers.dy(target_x)
		
		if evaluate_metrics:
			metrics = run_metrics(y, target_x, target_dx, target_dy, input)
			loss = config.model.loss(metrics)
		else:
			metrics = []
			loss = tf.constant(0.0)

		y_nhwc = image.tf_to_nhwc(y)
		input_nhwc = image.tf_to_nhwc(input)
		target_nhwc = image.tf_to_nhwc(train_target)
		
		if not config.model.input_enabled('gradients'):
			input_nhwc = tf.concat([input_nhwc, tf.zeros_like(input_nhwc), tf.zeros_like(input_nhwc)], axis=3)

	return {
		'losses': {'reconstruct': loss},
		'input': input,
		'metrics': metrics,
		'y': y,
		'y_uint8': image.tf_float_to_uint8(y_nhwc),
		'dx_uint8': image.tf_dx_to_uint8(y_nhwc),
		'dy_uint8': image.tf_dy_to_uint8(y_nhwc),
		'train_target': train_target,
	}
			
def trainable_model(gpu_inputs, define_model_fn, compile_optimizer=True):
	model_name = config.model.name
	warmup_factor = config.model.warmup_factor
	warmup_length = config.model.warmup_length
	rampdown_begin = config.model.rampdown_begin
	rampdown_stage_length = config.model.rampdown_stage_length
	rampdown_decay_per_stage = config.model.rampdown_decay_per_stage
	rampdown_func = config.model.rampdown_func
	
	model = {}
	model['train_ops'] = {}

	print("Building model '{}'.".format(model_name))
	
	with tf.device('/gpu:0'), tf.variable_scope('model', reuse=tf.AUTO_REUSE):
		# Compile the model.
		print("Compiling...", end="")
		with tf.name_scope('Network'):
			sys.stdout.flush()
			stime = time.time()
						
			network = define_model_fn(gpu_inputs, evaluate_metrics=compile_optimizer)
			model[0] = network

			print(" Done in {:0.1f} s.".format(time.time() - stime))
			
		batch_index = tf.get_variable("batch_index", dtype=tf.int64, trainable=False, initializer=tf.constant(0, dtype=tf.int64))

		# Compile the optimizer.
		if not network['losses']:
			print("No loss functions found. Skipping optimizer.")
			compile_optimizer = False

		if compile_optimizer:
			# Learning rate and batch indices.
			with tf.name_scope('Meta'):
				current_run_batch_index = tf.get_variable("current_run_batch_index", dtype=tf.int64, trainable=False, initializer=tf.constant(0, dtype=tf.int64))

				with tf.name_scope('update_batch_index'):
					update_batch_index = tf.assign(batch_index, batch_index + 1)
					update_current_run_batch_index = tf.assign(current_run_batch_index, current_run_batch_index + 1)

				initial_learning_rate = config.model.learning_rate
				
				learning_rate = tf.get_variable("learning_rate", dtype=tf.float32, trainable=False, initializer=tf.constant(initial_learning_rate, dtype=tf.float32))

				new_learning_rate = initial_learning_rate * (1.0 / warmup_factor) * warmup_factor**(tf.minimum(1.0, (1.0 + tf.cast(batch_index, tf.float32)) / float(warmup_length)))
				
				# Fade learning rate in after a potential restart to first safely evaluate Adam statistics.
				# Do 100 minibatches with no learning and then fade to current learning rate in 100 minibatches.
				new_learning_rate = new_learning_rate * tf.clip_by_value((tf.cast(current_run_batch_index, tf.float32) - 100.0) / 100.0, 0.0, 1.0)
				
				# Ramp-down at end.
				if rampdown_begin is not None:
					if rampdown_func == 'geometric':
						rampdown_stage = tf.ceil(tf.maximum(tf.cast(batch_index, tf.float32) - float(rampdown_begin), 0.0) / rampdown_stage_length)
					elif rampdown_func == 'exponential':
						rampdown_stage = tf.maximum(tf.cast(batch_index, tf.float32) - float(rampdown_begin), 0.0) / rampdown_stage_length
					else:
						raise Exception('Unknown ramp-down function.')

					new_learning_rate = new_learning_rate * (float(rampdown_decay_per_stage) ** rampdown_stage)
						
				with tf.control_dependencies([update_batch_index, update_current_run_batch_index]):
					update_learning_rate = tf.assign(learning_rate, new_learning_rate)

				tf.summary.scalar('learning_rate', learning_rate)
				tf.summary.scalar('batch_index', batch_index)
				tf.summary.scalar('current_run_batch_index', current_run_batch_index)	
			
			# Collect subnetwork names.
			subnetwork_names = []
			for key, val in network['losses'].items():
				print("Found loss '{}'.".format(key))
				subnetwork_names.append(key)

			# Create optimizers.
			with tf.name_scope('Optimizers'):
				optimizers = {}
				gradients = {}
				losses = {}
				
				for subnetwork_name in subnetwork_names:
					print("Compiling optimizer for '{}'...".format(subnetwork_name), end="")
					sys.stdout.flush()
					stime = time.time()

					optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.99, epsilon=1e-8)
					loss = network['losses'][subnetwork_name]
					variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/{}/'.format(subnetwork_name))
						
					losses[subnetwork_name] = loss
					optimizers[subnetwork_name] = optimizer
					gradients[subnetwork_name] = optimizer.compute_gradients(loss, var_list=variables)
				
					print(" Done in {:0.1f} s.".format(time.time() - stime))

				for subnetwork_name in subnetwork_names:
					model['train_ops'][subnetwork_name] = tf.group(optimizers[subnetwork_name].apply_gradients(gradients[subnetwork_name]))
			
		# Other meta variables.
		session_name = tf.get_variable("session_name", dtype=tf.string, trainable=False, initializer=tf.constant("no name"))
		
		summary_collection = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=tf.get_variable_scope().name)
		model['summaries'] = tf.summary.merge(summary_collection) if summary_collection else None
		
		model['metrics'] = network['metrics']
		model['name'] = model_name
		model['session_name'] = session_name
			
		saved_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/')
		saved_variables.extend([batch_index, session_name])

		model['saver'] = tf.train.Saver(saved_variables, max_to_keep=1)
		model['gpu_inputs'] = gpu_inputs

		if compile_optimizer:
			model['batch_index'] = batch_index
			model['update_learning_rate'] = update_learning_rate
			
			for subnetwork_name in subnetwork_names:
				tf.summary.scalar('loss_{}'.format(subnetwork_name), losses[subnetwork_name])
		
		# Calculate model size.
		model_variables = tf.trainable_variables(scope=tf.get_variable_scope().name + "/")
		model['parameter_count'] = np.sum([np.prod([dimension for dimension in variable.get_shape()]) for variable in model_variables])

	return model

			
def assign_inference_input(input_noisy):
	with tf.name_scope('assign_inference_input'):
		data = tf.placeholder(tf.float32, [None, None, None, None])
		kernel = tf.assign(input_noisy, data, validate_shape=False)

	return {'kernel': kernel, 'data': data}
			
def assign_training_input(input_noisy, input_clean):
	with tf.name_scope('minibatch'):
		data = tf.placeholder(tf.float16, [None, None, config.TOTAL_SIZE, config.TOTAL_SIZE])
	
	kernels = [
		tf.assign(input_noisy, tf.cast(data[:,0:config.INPUT_COUNT,:,:], dtype=tf.float32), validate_shape=False),
		tf.assign(input_clean, tf.cast(data[:,config.INPUT_COUNT:config.TOTAL_COUNT,:,:], dtype=tf.float32), validate_shape=False)
	]
	
	return {'kernel': kernels,
			'data': data}

