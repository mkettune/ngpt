import tensorflow as tf
import numpy as np


def save_npz(data, path):
	np.savez_compressed(path, [data])
	
def load_npz(path):
	npz_items = [x for x in np.load(path).items()]
	assert len(npz_items) == 1
		
	image = npz_items[0][1]
	if len(image.shape) == 4:
		image = image[0, :, :, :]
	return image

def save_png(image, path):
	import imageio
	
	image = np.clip(image, 0.0, 1.0)
	image = np.where(np.less_equal(image, 0.0031308), 12.92 * image, 1.055 * image ** (1.0/2.4) - 0.055)
	imageio.imwrite(path, (0.5 + 255.0 * image).astype(np.uint8))


def tf_to_nhwc(tensor):
	return tf.transpose(tensor, [0, 2, 3, 1])

def tf_evaluate_gradients(input):
	shape = tf.shape(input)
	h = shape[1]
	w = shape[2]
	
	input_dx = tf.pad(input[:, :, 1:w, :] - input[:, :, 0:w-1, :], tf.constant([[0,0], [0, 0], [0, 1], [0, 0]]))
	input_dy = tf.pad(input[:, 1:h, :, :] - input[:, 0:h-1, :, :], tf.constant([[0,0], [0, 1], [0, 0], [0, 0]]))
	
	return input_dx, input_dy

def tf_srgb_to_nonlinear(input):
	output = tf.where(tf.less_equal(input, 0.0031308), 12.92*input, 1.055*tf.pow(input, 1.0/2.4) - 0.055)
	return output

def tf_srgb_to_linear(input):
	output = tf.where(tf.less_equal(input, 0.04045), input/12.92, tf.pow((input+0.055)/1.055, 2.4))
	return output
	
def tf_uint8_to_float(tensor):
	image = tf.cast(tensor, tf.float32) / 255.0	
	image = tf.pow(image, 2.2)
	return image

def tf_float_to_uint8(tensor):
	image = tf.clip_by_value(tensor, 0.0, 1.0, name='clip_values')
	image = tf_srgb_to_nonlinear(image)
	
	image = tf.round(image * 255.0)
	image = tf.cast(image, tf.uint8)
	return image
	
def tf_dx_to_uint8(tensor):
	shape = tf.shape(tensor)
	h, w = shape[1], shape[2]
	return tf_float_to_uint8(0.5 + tf.pad(tensor[:, :, 1:w, :] - tensor[:, :, 0:w-1, :], tf.constant([[0,0], [0, 0], [0, 1], [0, 0]])))

def tf_dy_to_uint8(tensor):
	shape = tf.shape(tensor)
	h, w = shape[1], shape[2]
	return tf_float_to_uint8(0.5 + tf.pad(tensor[:, 1:h, :, :] - tensor[:, 0:h-1, :, :], tf.constant([[0,0], [0, 1], [0, 0], [0, 0]])))
