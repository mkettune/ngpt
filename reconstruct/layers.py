import pdb

import tensorflow as tf

from util import scope_name
import config
import image
import math
import numpy as np


### Derivatives.

def dx(input):
	shape = tf.shape(input)
	w = shape[3]
	return tf.pad(input[:, :, :, 1:w] - input[:, :, :, 0:w-1], tf.constant([[0,0], [0, 0], [0, 0], [0, 1]]))

def dy(input):
	shape = tf.shape(input)
	h = shape[2]
	return tf.pad(input[:, :, 1:h, :] - input[:, :, 0:h-1, :], tf.constant([[0,0], [0, 0], [0, 1], [0, 0]]))


### Variables.

def bias_variable(name, shape):
	variable = tf.get_variable(name=scope_name(name), shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())		
	return variable
	
def scale_variable(name, shape):
	variable = tf.get_variable(name=scope_name(name), shape=shape, dtype=tf.float32, initializer=tf.ones_initializer())
	return variable
	
def weight_variable(name, shape, initializer=None):
	if initializer is None:
		variable = tf.get_variable(name=scope_name(name), shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0))
	else:
		variable = tf.get_variable(name=scope_name(name), shape=shape, dtype=tf.float32, initializer=initializer)
	
	return variable


### Convolution operations.
	
def conv2d(input, output_feature_count, convolution_size, use_bias=True, use_relu=True, relu_factor=0.2):
	input_feature_count = input.get_shape()[1]

	convolution_width, convolution_height = convolution_size, convolution_size
		
	W = weight_variable("W", shape=[convolution_height, convolution_width, input_feature_count, output_feature_count])
	s = scale_variable("s", shape=[1, 1, 1, output_feature_count])			
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 2], keepdims=True)))

	if use_bias:
		b = bias_variable("b", shape=[output_feature_count])

	
	with tf.name_scope('layer'):
		layer = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
			
		if use_bias:
			layer = tf.nn.bias_add(layer, b, data_format='NCHW')
		
		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
			
	return layer


def denselist_conv2d(inputs, output_feature_count, convolution_size, use_bias=True, use_relu=True, relu_factor=0.2):
	'''Normal conv2d except that the input tensor is given as a list [T1, T2, T3] thought to be concatenated in the feature dimension,
	   and building the tensor explicitly is avoided.'''
	input_feature_count = sum(input.get_shape()[1].value for input in inputs)
	
	convolution_width, convolution_height = convolution_size, convolution_size
	
	W = weight_variable("W", shape=[convolution_height, convolution_width, input_feature_count, output_feature_count])
	s = scale_variable("s", shape=[1, 1, 1, output_feature_count])
		
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 2], keepdims=True)))

	if use_bias:
		b = bias_variable("b", shape=[output_feature_count])

	with tf.name_scope('layer'):
		range_begin = 0
		
		for i, input in enumerate(inputs):
			range_end = range_begin + input.get_shape()[1].value
			
			subresult = tf.nn.conv2d(input, W[:, :, range_begin:range_end, :], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
			
			if i == 0:
				layer = subresult
			else:
				layer += subresult
			
			range_begin = range_end
		
		if use_bias:
			layer = tf.nn.bias_add(layer, b, data_format='NCHW')
		
		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
			
	return layer
	
	
### Pooling operations.

def pool_conv(input, output_feature_count, use_relu=True, relu_factor=0.2):
	input_feature_count = input.get_shape()[1]

	W = weight_variable(name="W", shape=[2, 2, input_feature_count, output_feature_count])
	s = scale_variable("s", shape=[1, 1, 1, output_feature_count])
	
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 2], keepdims=True)))

	b = bias_variable(name="b", shape=[output_feature_count])
	
	with tf.name_scope('layer'):
		layer = tf.nn.conv2d(input, W, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')		
		layer = tf.nn.bias_add(layer, b, data_format='NCHW')
		
		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
			
	return layer

def denselist_pool_conv(inputs, output_feature_count, use_relu=True, relu_factor=0.2):
	'''Normal convolution pool except that the input tensor is given as a list [T1, T2, T3] thought to be concatenated in the feature dimension,
	   and building the tensor explicitly is avoided.'''
	  
	input_feature_count = sum(input.get_shape()[1].value for input in inputs)

	W = weight_variable(name="W", shape=[2, 2, input_feature_count, output_feature_count])
	s = scale_variable("s", shape=[1, 1, 1, output_feature_count])
	
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 2], keepdims=True)))
	
	b = bias_variable(name="b", shape=[output_feature_count])
	
	with tf.name_scope('layer'):
		range_begin = 0
		for i, input in enumerate(inputs):
			range_end = range_begin + input.get_shape()[1].value
			
			subresult = tf.nn.conv2d(input, W[:, :, range_begin:range_end, :], strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

			if i == 0:
				layer = subresult
			else:
				layer += subresult
			
			range_begin = range_end

		layer = tf.nn.bias_add(layer, b, data_format='NCHW')

		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
			
	return layer
	
def pool_average(input, scale=2):
	C = input.get_shape()[1].value
	assert C is not None

	with tf.name_scope('layer'):
		shape = tf.shape(input)
		N = shape[0]
		H = shape[2]
		W = shape[3]
		
		layer = tf.reshape(input, tf.stack([N, C, H//scale, scale, W//scale, scale]))
		layer = tf.reduce_mean(layer, axis=[3, 5])
		layer.set_shape([None, C, None, None])
	
	return layer

def pool_max(input, scale=2):
	C = input.get_shape()[1].value
	assert C is not None

	with tf.name_scope('layer'):
		shape = tf.shape(input)
		N = shape[0]
		H = shape[2]
		W = shape[3]
		
		layer = tf.reshape(input, tf.stack([N, C, H//scale, scale, W//scale, scale]))
		layer = tf.reduce_max(layer, axis=[3, 5])
		layer.set_shape([None, C, None, None])
	
	return layer

def pool_lanczos(input):
	'''Three-lobe 0.5x Lanczos downsampling.'''
	K = 2

	I = input

	C = input.get_shape()[1].value
	assert C is not None
	
	shape = tf.shape(I)
	N, H, W = shape[0], shape[2], shape[3]

	# Lanczos cross-correlation kernel.
	K0p5 = np.asarray([np.sinc((i-0.5)/float(K))*np.sinc((i-0.5)/(2.0 * K)) for i in range(-2*K+1, 2*K+1)], dtype=np.float32)
	K0p5 = K0p5 / sum(K0p5)

	# Lanczos in Y.
	I = tf.pad(I, [(0, 0), (0, 0), (3, 3), (3, 3)], 'reflect')
	I0 = I[:, :, 0:H  :2, :]
	I1 = I[:, :, 1:H+1:2, :]
	I2 = I[:, :, 2:H+2:2, :]
	I3 = I[:, :, 3:H+3:2, :]	
	I4 = I[:, :, 4:H+4:2, :]	
	I5 = I[:, :, 5:H+5:2, :]	
	I6 = I[:, :, 6:H+6:2, :]	
	I7 = I[:, :, 7:H+7:2, :]	
	I = (I0 * K0p5[0] + I1 * K0p5[1] + I2 * K0p5[2] + I3 * K0p5[3] +
	     I4 * K0p5[4] + I5 * K0p5[5] + I6 * K0p5[6] + I7 * K0p5[7])

	# Lanczos in X.
	I0 = I[:, :, :, 0:W  :2]
	I1 = I[:, :, :, 1:W+1:2]
	I2 = I[:, :, :, 2:W+2:2]
	I3 = I[:, :, :, 3:W+3:2]	
	I4 = I[:, :, :, 4:W+4:2]	
	I5 = I[:, :, :, 5:W+5:2]	
	I6 = I[:, :, :, 6:W+6:2]	
	I7 = I[:, :, :, 7:W+7:2]	
	I = (I0 * K0p5[0] + I1 * K0p5[1] + I2 * K0p5[2] + I3 * K0p5[3] +
	     I4 * K0p5[4] + I5 * K0p5[5] + I6 * K0p5[6] + I7 * K0p5[7])
	
	I.set_shape([None, C, None, None])
	return I

	
### Unpooling.

def unpool_conv(input, output_feature_count, use_relu=True, relu_factor=0.2):
	input_feature_count = input.get_shape()[1]
	
	W = weight_variable("W", shape=[2, 2, output_feature_count, input_feature_count])
	s = scale_variable("s", shape=[1, 1, output_feature_count, 1])
	
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 3], keepdims=True)))

	b = bias_variable("b", shape=[output_feature_count])
	
	with tf.name_scope('layer'):
		dynamic_input_shape = tf.shape(input)
		batch_size = dynamic_input_shape[0]
		height = dynamic_input_shape[2]
		width = dynamic_input_shape[3]
		layer_shape = tf.stack([batch_size, output_feature_count, 2 * height, 2 * width])
		
		layer = tf.nn.conv2d_transpose(input, W, output_shape=layer_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

		layer = tf.nn.bias_add(layer, b, data_format='NCHW')
		
		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
			
	return layer
	

def denselist_unpool_conv(inputs, output_feature_count, use_relu=True, relu_factor=0.2):
	'''Normal convolution transpose unpooling except that the input tensor is given as a list [T1, T2, T3] thought to be concatenated in the feature dimension,
	   and building the tensor explicitly is avoided.'''
	
	input_feature_count = sum(input.get_shape()[1].value for input in inputs)
	
	W = weight_variable("W", shape=[2, 2, output_feature_count, input_feature_count])
	s = scale_variable("s", shape=[1, 1, output_feature_count, 1])
	
	W0_norm = math.sqrt(2.0)
	W = W * (s * W0_norm / tf.sqrt(1e-8 + tf.reduce_sum(tf.square(W), axis=[0, 1, 3], keepdims=True)))

	b = bias_variable("b", shape=[output_feature_count])

	dynamic_input_shape = tf.shape(inputs[0])
	batch_size = dynamic_input_shape[0]
	height = dynamic_input_shape[2]
	width = dynamic_input_shape[3]
	
	with tf.name_scope('layer'):	
		layer_shape = tf.stack([batch_size, output_feature_count, 2 * height, 2 * width])

		range_begin = 0
		
		for i, input in enumerate(inputs):
			range_end = range_begin + input.get_shape()[1].value
			
			subresult = tf.nn.conv2d_transpose(input, W[:, :, :, range_begin:range_end], output_shape=layer_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

			if i == 0:
				layer = subresult
			else:
				layer += subresult

			range_begin = range_end
		
		layer = tf.nn.bias_add(layer, b, data_format='NCHW')

		if use_relu:
			layer = tf.nn.leaky_relu(layer, relu_factor)
	
	
	return layer
	
def unpool_nearest(input, scale=2):
	C = input.get_shape()[1].value
	assert C is not None
	
	with tf.name_scope('layer'):
		shape = tf.shape(input)
		N = shape[0]
		H = shape[2]
		W = shape[3]
		
		# New.
		layer = tf.reshape(input, [N, C, H, 1, W, 1])
		layer = tf.tile(layer, [1, 1, 1, scale, 1, scale])
		layer = tf.reshape(layer, [N, C, scale*H, scale*W])
		layer.set_shape([None, C, None, None])

	return layer
	
def unpool_lanczos(input):
	'''Three-lobe 2x Lanczos upsampling.'''

	I = input

	C = input.get_shape()[1].value
	assert C is not None
	
	shape = tf.shape(I)
	N, H, W = shape[0], shape[2], shape[3]

	# Lanczos cross-correlation kernel.
	K0p25 = np.asarray([np.sinc(i-0.25)*np.sinc((i-0.25)/2.0) for i in range(-1, 2+1)], dtype=np.float32)
	K0p25 = K0p25 / sum(K0p25)

	K0p75 = np.asarray([np.sinc(i-0.75)*np.sinc((i-0.75)/2.0) for i in range(-1, 2+1)], dtype=np.float32)
	K0p75 = K0p75 / sum(K0p75)

	# Lanczos in Y.
	I = tf.pad(I, [(0, 0), (0, 0), (2, 2), (2, 2)], 'reflect')
	I0 = I[:, :, 0:H+1, :]
	I1 = I[:, :, 1:H+2, :]
	I2 = I[:, :, 2:H+3, :]
	I3 = I[:, :, 3:H+4, :]
	
	I = tf.stack([
		I0 * K0p25[0] + I1 * K0p25[1] + I2 * K0p25[2] + I3 * K0p25[3],
		I0 * K0p75[0] + I1 * K0p75[1] + I2 * K0p75[2] + I3 * K0p75[3]
	], axis=3)
	I = tf.reshape(I, [N, C, 2*H+2, W+4])
	
	# Lanczos in X.
	I0 = I[:, :, :, 0:W+1]
	I1 = I[:, :, :, 1:W+2]
	I2 = I[:, :, :, 2:W+3]
	I3 = I[:, :, :, 3:W+4]
	
	I = tf.stack([
		I0 * K0p25[0] + I1 * K0p25[1] + I2 * K0p25[2] + I3 * K0p25[3],
		I0 * K0p75[0] + I1 * K0p75[1] + I2 * K0p75[2] + I3 * K0p75[3]
	], axis=4)
	I = tf.reshape(I, [N, C, 2*H+2, 2*W+2])
	
	I = I[:, :, 1:2*H+1, 1:2*W+1]
	I.set_shape([None, C, None, None])

	return I
	
def unpool_bilinear(input):
	C = input_feature_count = input.get_shape()[1]
	assert C is not None

	I = input
	
	shape = tf.shape(I)
	N, H, W = shape[0], shape[2], shape[3]

	# Cross-correlation kernels.
	K0p25 = np.asarray([0.75, 0.25], dtype=np.float32)
	K0p75 = np.asarray([0.25, 0.75], dtype=np.float32)
	K0p75 = np.tile(np.reshape(K0p75, [1, 1, 2, 1]), [1, input_feature_count.value, 1, 1])
	
	# Resample in Y.
	I = tf.pad(I, [(0, 0), (0, 0), (1, 1), (1, 1)], 'reflect')

	I0 = I[:, :, 0:H+1, :]
	I1 = I[:, :, 1:H+2, :]
	
	I = tf.stack([I0 * 0.75 + I1 * 0.25, I0 * 0.25 + I1 * 0.75], axis=3)
	I = tf.reshape(I, [N, C, 2*H+2, W+2])

	# Resample in X.
	I0 = I[:, :, :, 0:W+1]
	I1 = I[:, :, :, 1:W+2]

	I = tf.stack([I0 * 0.75 + I1 * 0.25, I0 * 0.25 + I1 * 0.75], axis=4)
	I = tf.reshape(I, [N, C, 2*H+2, 2*W+2])	
	
	I = I[:, :, 1:2*H+1, 1:2*W+1]

	I.set_shape([None, C, None, None])

	return I

	
### Other layers.

def concat(inputs):	
	layer = tf.concat([input for input in inputs], axis=1)
	return layer

def srgb_to_nonlinear(input):
	output = tf.where(tf.less_equal(input, 0.0031308), 12.92*input, 1.055*tf.pow(input, 1.0/2.4) - 0.055)
	return output
	
def srgb_to_linear(input):
	output = tf.where(tf.less_equal(input, 0.04045), input/12.92, tf.pow((input+0.055)/1.055, 2.4))
	return output