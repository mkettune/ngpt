import numpy as np
import multiprocessing
import random
import sys
import copy
import time
import pdb
import gc

import darc
import shared

import config
import util


### Padding functions.
	 
def mirrorPadPrimal(image, pad_x, pad_y):
	if config.PAD_WIDTH == 0:
		return image
		
	# Add padding.
	result = np.pad(image, [(0, 0), pad_y, pad_x], 'reflect')
	return result

def mirrorPadOdd(image, pad_x, pad_y):
	if config.PAD_WIDTH == 0:
		return image
		
	# Add padding.
	result = np.pad(image, [(0, 0), pad_y, pad_x], 'reflect', reflect_type='odd')
	return result
	
def mirrorPadDx(image, pad_x, pad_y):
	if pad_x[0] == pad_x[1] == pad_y[0] == pad_y[1] == 0:
		image = np.concatenate([
			image[:, :, 0:-1],
			np.zeros_like(image[:, :, 0:1])
		], axis=2)

		return image
		
	# Skip last row/column.
	image = image[:, :, 0:-1]

	# Anti-symmetric padding in diff direction.
	sub_images = []
	if pad_x[0] > 0: sub_images.append(-image[:, :, pad_x[0]-1::-1])
	sub_images.append(image)
	if pad_x[1] > 0: sub_images.append(-image[:, :, -1:-pad_x[1]-1:-1])
	sub_images.append(np.zeros_like(image[:, :, 0:1]))
	image = np.concatenate(sub_images, axis=2) 	
	
	if pad_y[0] != 0 or pad_y[1] != 0:
		result = np.pad(image, [(0, 0), pad_y, (0, 0)], 'reflect') # Reflect padding in the other direction.
	else:
		result = image
		
	return result
	
def mirrorPadDy(image, pad_x, pad_y):
	if pad_x[0] == pad_x[1] == pad_y[0] == pad_y[1] == 0:
		image = np.concatenate([
			image[:, 0:-1, :],
			np.zeros_like(image[:, 0:1, :])
		], axis=1)

		return image
	
	# Skip last row/column.
	image = image[:, 0:-1, :]

	# Anti-symmetric padding in diff direction.
	sub_images = []
	if pad_y[0] > 0: sub_images.append(-image[:, pad_y[0]-1::-1, :])
	sub_images.append(image)
	if pad_y[1] > 0: sub_images.append(-image[:, -1:-pad_y[1]-1:-1, :])
	sub_images.append(np.zeros_like(np.zeros_like(image[:, 0:1, :])))
	image = np.concatenate(sub_images, axis=1)
	
	if pad_x[0] != 0 or pad_x[1] != 0:
		result = np.pad(image, [(0, 0), (0, 0), pad_x], 'reflect') # Reflect padding in the other direction.
	else:
		result = image

	return result
	
def mirrorPadVarDx(image, pad_x, pad_y):
	if pad_x[0] == pad_x[1] == pad_y[0] == pad_y[1] == 0:
		image = np.concatenate([
			image[:, :, 0:-1],
			np.zeros_like(image[:, :, 0:1])
		], axis=2)

		return image
		
	# Skip last row/column.
	image = image[:, :, 0:-1]

	# Symmetric padding in diff direction.
	sub_images = []
	if pad_x[0] > 0: sub_images.append(image[:, :, pad_x[0]-1::-1])
	sub_images.append(image)
	if pad_x[1] > 0: sub_images.append(image[:, :, -1:-pad_x[1]-1:-1])
	sub_images.append(np.zeros_like(image[:, :, 0:1]))
	image = np.concatenate(sub_images, axis=2) 

	
	if pad_y[0] != 0 or pad_y[1] != 0:
		result = np.pad(image, [(0, 0), pad_y, (0, 0)], 'reflect') # Reflect padding in the other direction.
	else:
		result = image
		
	return result

def mirrorPadVarDy(image, pad_x, pad_y):
	if pad_x[0] == pad_x[1] == pad_y[0] == pad_y[1] == 0:
		image = np.concatenate([
			image[:, 0:-1, :],
			np.zeros_like(image[:, 0:1, :])
		], axis=1)

		return image
	
	# Skip last row/column.
	image = image[:, 0:-1, :]

	# Symmetric padding in diff direction.
	sub_images = []
	if pad_y[0] > 0: sub_images.append(image[:, pad_y[0]-1::-1, :])
	sub_images.append(image)
	if pad_y[1] > 0: sub_images.append(image[:, -1:-pad_y[1]-1:-1, :])
	sub_images.append(np.zeros_like(np.zeros_like(image[:, 0:1, :])))
	image = np.concatenate(sub_images, axis=1)
	
	if pad_x[0] != 0 or pad_x[1] != 0:
		result = np.pad(image, [(0, 0), (0, 0), pad_x], 'reflect') # Reflect padding in the other direction.
	else:
		result = image

	return result

	
### Dataset readers.

class MultiFileArchive:
	'''Virtually joins multiple Archives into one.'''
	def __init__(self, paths, validation=False):
		assert paths
		
		self.path = paths
		self.validation = validation
		self.dimensions = config.TOTAL_COUNT
		
		self.archives = []

		self.image_count = 0
		
		for path in self.path:
			archive = Archive(path, validation)
			self.archives.append(archive)
		
			self.image_count += archive.image_count
		
		self.crop_count = self.archives[0].crop_count
			
	def readCrop(self, index):
		# Return from the correct archive.
		for archive in self.archives:
			if index < archive.image_count:
				return archive.readCrop(index)
			index -= archive.image_count
		
		raise Exception("Crop index out of range.")
	
	 
class Archive:
	'''Interface for reading a Darc database.'''
	def __init__(self, path, validation=False):
		self.path = path
		self.darc = darc.DataArchive(path, 'r')
		
		assert len(self.darc) > 0 # Empty archive?
		
		self.crop_count = self.darc[0].shape[0] - 1
		self.image_count = len(self.darc) * self.crop_count
		
		self.validation = validation
		self.dimensions = config.TOTAL_COUNT
	
	def readCrop(self, index, special=None):		
		crop_instance = index % self.crop_count
		
		if special == 'USE_REFERENCE': # Return the reference image.
			crop_instance = self.crop_count			
		
		# Choose which image to return.
		index = index // self.crop_count
		input_image = self.darc[index]
		
		def fix_variance(buffer, zero_to_none=True):
			buffer[np.isinf(buffer)] = -1.0
			return buffer
		
		def toSlice(x):
			return slice(x[0], x[1])
		
		# Define augmentation.
		flip_x, flip_y, swap_xy = False, False, False
		rgb_order = np.asarray([0, 1, 2])
		brightness = 1.0
		color_multiplier = np.tile(1.0, [1, 1, 3])
		padding_offset_x = 0
		padding_offset_y = 0
		
		# Choose augmentation parameters.
		if not self.validation:
			if config.model.augmentation_enabled('flip_x'):
				flip_x = random.random() < 0.5
			
			if config.model.augmentation_enabled('flip_y'):
				flip_y = random.random() < 0.5
			
			if config.model.augmentation_enabled('swap_xy'):
				swap_xy = random.random() < 0.5
			
			if config.model.augmentation_enabled('permute_rgb'):
				rgb_order = np.random.permutation(3)
			
			if config.model.vary_padding > 0:
				padding_offset_x = random.randint(-config.model.vary_padding, config.model.vary_padding)
				padding_offset_y = random.randint(-config.model.vary_padding, config.model.vary_padding)
			
			if config.model.augmentation_enabled('brightness'):
				brightness = np.random.lognormal(size=1) / np.exp(0.5) # Log-normal with mean = 1.0.

			if config.model.augmentation_enabled('color_multiply'):
				# Color multipliers for channels such that they don't change brightness on average.
				# p(u,v,w | u+v+w=1) = c
				u = random.random()
				v = random.random()
				if u + v > 1.0:
					u, v = 1.0 - v, 1.0 - u
				w = 1.0 - u - v
				
				color_multiplier = np.asarray([3.0 * u, 3.0 * v, 3.0 * w], dtype=np.float32)
				color_multiplier = color_multiplier.reshape([1, 1, 3])
				
		
		### Define the flipping functions for all inputs.
		
		def flip_primal(tensor):
			if flip_x:
				if flip_y:
					return tensor[::-1, ::-1, :]
				else:
					return tensor[:, ::-1, :]
			else:
				if flip_y:
					return tensor[::-1, :, :]
				else:
					return tensor
        
		def flip_dx(tensor):
			H, W, C = tensor.shape
			
			if flip_x:
				if flip_y:
					return np.pad(-tensor[::-1, W-2::-1, :], [(0,0), (0,1), (0,0)], mode='symmetric')
				else:
					return np.pad(-tensor[:, W-2::-1, :], [(0,0), (0,1), (0,0)], mode='symmetric')
			else:
				if flip_y:
					return tensor[::-1, :, :]
				else:
					return tensor
        
		def flip_dy(tensor):
			H, W, C = tensor.shape
			
			if flip_x:
				if flip_y:
					return np.pad(-tensor[H-2::-1, ::-1, :], [(0,1), (0,0), (0,0)], mode='symmetric')
				else:
					return tensor[:, ::-1, :]
			else:
				if flip_y:
					return np.pad(-tensor[H-2::-1, :, :], [(0,1), (0,0), (0,0)], mode='symmetric')
				else:
					return tensor
        
		def flip_var_primal(tensor):
			return flip_primal(tensor)
        
		def flip_var_dx(tensor):
			H, W, C = tensor.shape
			
			if flip_x:
				if flip_y:
					return np.pad(tensor[::-1, W-2::-1, :], [(0,0), (0,1), (0,0)], mode='symmetric')
				else:
					return np.pad(tensor[:, W-2::-1, :], [(0,0), (0,1), (0,0)], mode='symmetric')
			else:
				if flip_y:
					return tensor[::-1, :, :]
				else:
					return tensor
        
		def flip_var_dy(tensor):
			H, W, C = tensor.shape
			
			if flip_x:
				if flip_y:
					return np.pad(tensor[H-2::-1, ::-1, :], [(0,1), (0,0), (0,0)], mode='symmetric')
				else:
					return tensor[:, ::-1, :]
			else:
				if flip_y:
					return np.pad(tensor[H-2::-1, :, :], [(0,1), (0,0), (0,0)], mode='symmetric')
				else:
					return tensor
			
		def flip_normal(tensor):
			H, W, C = tensor.shape
			
			if flip_x:
				if flip_y:
					tensor[:, :, 0] = -tensor[:, :, 0]
					tensor[:, :, 1] = -tensor[:, :, 1]
					return tensor[::-1, ::-1, :]
				else:
					tensor[:, :, 0] = -tensor[:, :, 0]
					return tensor[:, ::-1, :]
			else:
				if flip_y:
					tensor[:, :, 1] = -tensor[:, :, 1]
					return tensor[::-1, :, :]
				else:
					return tensor
        
		def flip_var_normal(tensor):
			return flip_primal(tensor)
        
		def flip_albedo(tensor):
			return flip_primal(tensor)
        
		def flip_var_albedo(tensor):
			return flip_primal(tensor)
		
		def flip_depth(tensor):
			return flip_primal(tensor)
		
		def flip_var_depth(tensor):
			return flip_primal(tensor)
			
		def flip_diffuse(tensor):
			return flip_primal(tensor)
		
		def flip_var_diffuse(tensor):
			return flip_primal(tensor)
			
		def flip_specular(tensor):
			return flip_primal(tensor)
		
		def flip_var_specular(tensor):
			return flip_primal(tensor)
			
		
		### Define the actual augmentation functions.
		
		def augment_primal(tensor, var_tensor, rgb_order):
			# Reorder color channels.
			np.take(tensor, rgb_order, axis=2, out=tensor)
        
			# Flip x and y.
			tensor = flip_primal(tensor)
			
			# Swap x and y.
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			colored_primal = color_multiplier * tensor
			
			brightened_primal = colored_primal * brightness
			
			# Handle variance.
			if var_tensor is not None:
				var_tensor = flip_var_primal(var_tensor)
			
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
				
				good_mask = (var_tensor >= 0.0)
				var_tensor[good_mask] = np.log(1.0 + (brightness**2) * var_tensor[good_mask])

			tensor = np.log(1.0 + brightened_primal)
			
			return tensor, var_tensor
		
		def augment_dx_dy(tensor_dx, tensor_dy, var_tensor_dx, var_tensor_dy, rgb_order):			
			np.take(tensor_dx, rgb_order, axis=2, out=tensor_dx)
			np.take(tensor_dy, rgb_order, axis=2, out=tensor_dy)
			
			tensor_dx = flip_dx(tensor_dx)
			tensor_dy = flip_dy(tensor_dy)
        
			if swap_xy:
				tensor_dx, tensor_dy = tensor_dy, tensor_dx
				tensor_dx = np.transpose(tensor_dx, axes=(1,0,2))
				tensor_dy = np.transpose(tensor_dy, axes=(1,0,2))
			
			colored_dx = color_multiplier * tensor_dx
			colored_dy = color_multiplier * tensor_dy
        
			dx = colored_dx * brightness
			dy = colored_dy * brightness
			
			dx = np.sign(dx) * np.log(1.0 + np.abs(dx))
			dy = np.sign(dy) * np.log(1.0 + np.abs(dy))
			
			if var_tensor_dx is not None and var_tensor_dy is not None:				
				var_tensor_dx = flip_var_dx(var_tensor_dx)
				var_tensor_dy = flip_var_dy(var_tensor_dy)
        
				if swap_xy:
					var_tensor_dx, var_tensor_dy = var_tensor_dy, var_tensor_dx
					var_tensor_dx = np.transpose(var_tensor_dx, axes=(1,0,2))
					var_tensor_dy = np.transpose(var_tensor_dy, axes=(1,0,2))
				
				good_mask = (var_tensor_dx >= 0.0)
				var_tensor_dx[good_mask] = np.log(1.0 + (brightness**2) * var_tensor_dx[good_mask])

				good_mask = (var_tensor_dy >= 0.0)
				var_tensor_dy[good_mask] = np.log(1.0 + (brightness**2) * var_tensor_dy[good_mask])
			
			return dx, dy, var_tensor_dx, var_tensor_dy
		
		def augment_normal(tensor, var_tensor=None):
			tensor = flip_normal(tensor)
			
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
				
				dummy = tensor[:, :, 0].copy()
				tensor[:, :, 0] = tensor[:, :, 1]
				tensor[:, :, 1] = dummy
			
			# Handle variance.
			if var_tensor is not None:
				var_tensor = flip_var_normal(var_tensor)
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
        			
			return tensor, var_tensor
		
		def augment_albedo(tensor, var_tensor, rgb_order):
			# Reorder color channels.
			tensor = np.take(tensor, rgb_order, axis=2)
        
			# Flip x and y.
			tensor = flip_albedo(tensor)
			
			# Swap x and y.
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			
			# Flip and swap variances too.
			if var_tensor is not None:
				var_tensor = flip_var_albedo(var_tensor)
			
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
			
			augmented_albedo = tensor * color_multiplier
			
			# Apply color multiplication to variance.
			if var_tensor is not None:
				# See above.
				var_scale = np.sum(np.square(augmented_albedo), axis=2, keepdims=True) / (1e-10 + np.sum(np.square(tensor), axis=2, keepdims=True))
				
				good_mask = (var_tensor >= 0.0)
				var_tensor[good_mask] = var_scale[good_mask] * var_tensor[good_mask]
				
			
			return augmented_albedo, var_tensor # Notice: this may make albedos greater than 1 but it's ok as it's only for training
                
		def augment_depth(tensor, var_tensor):
			tensor = flip_depth(tensor)
			
			# Swap x and y.
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			# Handle variance.
			if var_tensor is not None:
				var_tensor = flip_var_depth(var_tensor)
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
			
			if not self.validation:
				# Scale maximum depth to around 1 with some variance so that 1.000 will not be an outlier.
				scale = np.random.uniform(0.1, 1.1) / (1e-10 + np.amax(tensor))
			else:
				scale = 1.0 / (1e-10 + np.amax(tensor))
			
			tensor *= scale
			
			if var_tensor is not None:
				good_mask = (var_tensor >= 0.0)
				
				var_scale = scale * scale
				var_tensor[good_mask] = var_scale * var_tensor[good_mask]
			
			return tensor, var_tensor
			
		def augment_diffuse(tensor, var_tensor, rgb_order):
			# Reorder color channels.
			np.take(tensor, rgb_order, axis=2, out=tensor)
        
			tensor = flip_diffuse(tensor)
			
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			primal = brightness * color_multiplier * tensor
			
			# Handle variance.
			if var_tensor is not None:
				var_tensor = flip_var_diffuse(var_tensor)
			
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
			
				good_mask = (var_tensor >= 0.0)
				var_tensor[good_mask] = np.log(1.0 + (brightness**2) * var_tensor[good_mask])

			tensor = np.log(1.0 + primal)
			return tensor, var_tensor
		
		def augment_specular(tensor, var_tensor, rgb_order):
			# Reorder color channels.
			np.take(tensor, rgb_order, axis=2, out=tensor)
        
			tensor = flip_specular(tensor)
			
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			primal = brightness * color_multiplier * tensor
			
			# Handle variance.
			if var_tensor is not None:
				var_tensor = flip_var_diffuse(var_tensor)
			
				if swap_xy:
					var_tensor = np.transpose(var_tensor, axes=(1,0,2))
			
				good_mask = (var_tensor >= 0.0)
				var_tensor[good_mask] = np.log(1.0 + (brightness**2) * var_tensor[good_mask])

			tensor = np.log(1.0 + primal)
			return tensor, var_tensor
		
		def augment_clean(tensor, rgb_order):			
			# Reorder color channels.
			np.take(tensor, rgb_order, axis=2, out=tensor)
        
			# Flip x and y.
			tensor = flip_primal(tensor)
			
			# Swap x and y.
			if swap_xy:
				tensor = np.transpose(tensor, axes=(1,0,2))
			
			colored_primal = color_multiplier * tensor
			
			brightened_primal = colored_primal * brightness
			
			tensor = brightened_primal
			return tensor #colored_primal * brightness, var_tensor
		
		
		results = []
		
		# Read all required buffers.
		primal_in, dx_in, dy_in, var_primal_in, var_dx_in, var_dy_in = None, None, None, None, None, None
		albedo_in, normal_in, depth_in = None, None, None
		var_albedo_in, var_normal_in, var_depth_in = None, None, None
		diffuse_in, specular_in, var_diffuse_in, var_specular_in = None, None, None, None # Note: Not used by our network.
		
		primal_in = input_image[crop_instance, :, :, toSlice(config.IND_PRIMAL_COLOR)].astype(np.float32)
		if config.model.input_enabled('gradients'):
			dx_in = input_image[crop_instance, :, :, toSlice(config.IND_DX_COLOR)].astype(np.float32)
			dy_in = input_image[crop_instance, :, :, toSlice(config.IND_DY_COLOR)].astype(np.float32)
		
		if config.model.input_enabled('variances'):
			if config.model.input_enabled('gradients'):
				var_dx_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_DX)].astype(np.float32)
				var_dy_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_DY)].astype(np.float32)
				
				var_dx_in = fix_variance(var_dx_in, zero_to_none=True)
				var_dy_in = fix_variance(var_dy_in, zero_to_none=True)
			
			var_primal_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_PRIMAL)].astype(np.float32)
			var_primal_in = fix_variance(var_primal_in, zero_to_none=True)
		
		if config.model.input_enabled('albedo'):
			albedo_in = input_image[crop_instance, :, :, toSlice(config.IND_ALBEDO)].astype(np.float32)
		if config.model.input_enabled('normal'):
			normal_in = input_image[crop_instance, :, :, toSlice(config.IND_NORMAL)].astype(np.float32)
		if config.model.input_enabled('depth'):
			depth_in = input_image[crop_instance, :, :, toSlice(config.IND_DEPTH)].astype(np.float32)
	
		# Note: Not used by our network.
		if config.model.input_enabled('diffuse'):
			diffuse_in = input_image[crop_instance, :, :, toSlice(config.IND_DIFFUSE)].astype(np.float32)
		if config.model.input_enabled('specular'):
			specular_in = input_image[crop_instance, :, :, toSlice(config.IND_SPECULAR)].astype(np.float32)			
			
		if config.model.input_enabled('variances'):			
			if config.model.input_enabled('albedo'):
				var_albedo_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_ALBEDO)].astype(np.float32)
				var_albedo_in = fix_variance(var_albedo_in, zero_to_none=False)
			if config.model.input_enabled('normal'):
				var_normal_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_NORMAL)].astype(np.float32)
				var_normal_in = fix_variance(var_normal_in, zero_to_none=False)
			if config.model.input_enabled('depth'):
				var_depth_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_DEPTH)].astype(np.float32)
				var_depth_in = fix_variance(var_depth_in, zero_to_none=True)
			if config.model.input_enabled('diffuse'):
				var_diffuse_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_DIFFUSE)].astype(np.float32)
				var_diffuse_in = fix_variance(var_diffuse_in)
			if config.model.input_enabled('specular'):
				var_specular_in = input_image[crop_instance, :, :, toSlice(config.IND_VAR_SPECULAR)].astype(np.float32)			
				var_specular_in = fix_variance(var_specular_in)

		clean_in = input_image[self.crop_count, :, :, toSlice(config.IND_PRIMAL_COLOR)].astype(np.float32)
        
		
		# Augment and prepare the buffers.
		primal, dx, dy, var_primal, var_dx, var_dy = None, None, None, None, None, None
		albedo, normal, depth = None, None, None
		var_albedo, var_normal, var_depth = None, None, None
		diffuse, specular, var_diffuse, var_specular = None, None, None, None

		primal, var_primal = augment_primal(primal_in, var_primal_in, rgb_order)
		
		if config.model.input_enabled('gradients'):
			dx, dy, var_dx, var_dy = augment_dx_dy(dx_in, dy_in, var_dx_in, var_dy_in, rgb_order)
        
		if config.model.input_enabled('albedo'):
			albedo, var_albedo = augment_albedo(albedo_in, var_albedo_in, rgb_order)
			
		if config.model.input_enabled('normal'):
			normal, var_normal = augment_normal(normal_in, var_normal_in)
			
		if config.model.input_enabled('depth'):
			depth, var_depth = augment_depth(depth_in, var_depth_in)
			
		if config.model.input_enabled('diffuse'):
			diffuse, var_diffuse = augment_diffuse(diffuse_in, var_diffuse_in, rgb_order)

		if config.model.input_enabled('specular'):
			specular, var_specular = augment_specular(specular_in, var_specular_in, rgb_order)
		
		clean = augment_clean(clean_in, rgb_order)
		
		
		# Transpose all to NCHW.
		transpose = lambda x: np.transpose(x, [2, 0, 1]) if x is not None else None
		
		primal, var_primal = transpose(primal), transpose(var_primal)
		dx, var_dx = transpose(dx), transpose(var_dx)
		dy, var_dy = transpose(dy), transpose(var_dy)
		albedo, var_albedo = transpose(albedo), transpose(var_albedo)
		normal, var_normal = transpose(normal), transpose(var_normal)
		depth, var_depth = transpose(depth), transpose(var_depth)
		diffuse, var_diffuse = transpose(diffuse), transpose(var_diffuse)
		specular, var_specular = transpose(specular), transpose(var_specular)
		clean = transpose(clean)
        
		
		# Pad enabled buffers.
		pad_x = (config.PAD_WIDTH + padding_offset_x, config.PAD_WIDTH - padding_offset_x)
		pad_y = (config.PAD_WIDTH + padding_offset_y, config.PAD_WIDTH - padding_offset_y)
		
		pad = lambda buffer, f: f(buffer, pad_x, pad_y) if buffer is not None else None
		
		results.append(pad(primal, mirrorPadPrimal))
		results.append(pad(dx, mirrorPadDx))
		results.append(pad(dy, mirrorPadDy))
		results.append(pad(var_primal, mirrorPadPrimal))
		results.append(pad(var_dx, mirrorPadVarDx))
		results.append(pad(var_dy, mirrorPadVarDy))
		results.append(pad(albedo, mirrorPadPrimal))
		results.append(pad(normal, mirrorPadPrimal))
		results.append(pad(depth, mirrorPadOdd))
		results.append(pad(var_albedo, mirrorPadPrimal))
		results.append(pad(var_normal, mirrorPadPrimal))
		results.append(pad(var_depth, mirrorPadPrimal))
		results.append(pad(diffuse, mirrorPadPrimal))
		results.append(pad(specular, mirrorPadPrimal))
		results.append(pad(var_diffuse, mirrorPadPrimal))
		results.append(pad(var_specular, mirrorPadPrimal))
		results.append(pad(clean, mirrorPadPrimal))
        
		# Retain only the enabled buffers.
		results = [x for x in results if x is not None]
		
		# Note: The order must be exactly the same as defined in config.py.
		
		# Concatenate the enabled features.
		return np.concatenate(results, axis=0)

		
class BatchDivider:
	'''Divides an epoch defined by an Archive into minibatches.'''
	def __init__(self, archive):
		self.archive = archive
		
		self.epoch_index = 0
		self.remaining = set(range(self.archive.image_count))
	
	def initEpoch(self):
		self.epoch_index += 1
		self.remaining = set(range(self.archive.image_count))

	def produceBatchIndices(self, number):
		result = []
		while len(result) < number:
			count = min(len(self.remaining), number - len(result))

			L = random.sample(self.remaining, count)
			self.remaining.difference_update(L)
			
			result.extend(L)
			
			if not self.remaining:
				self.initEpoch()

		return result
		
	def constructBatch(self, indices):
		count = len(indices)
		
		data = np.empty([count, self.archive.dimensions, config.TOTAL_SIZE, config.TOTAL_SIZE], dtype=np.float16)
			
		for i in range(count):
			data[i, :, :, :] = self.archive.readCrop(indices[i])
				
		return data

class DigitReversingBatchDivider(BatchDivider):
	'''Divides an epoch defined by an Archive into minibatches in reversed bit order.'''
	def __init__(self, archive):
		super().__init__(archive)
		
		self.full_set = util.digit_reversed_range(self.archive.image_count, 2)
		self.current_index = 0
	
	def initEpoch(self):
		self.current_index = 0
		self.epoch_index += 1
		
	def produceBatchIndices(self, number):
		result = []
		while len(result) < number:
			count = min(len(self.full_set) - self.current_index, number - len(result))

			L = self.full_set[self.current_index : self.current_index + count]			
			self.current_index += count
			
			result.extend(L)
			
			if self.current_index >= len(self.full_set):
				self.initEpoch()

		return result

		
def open_archive(paths, validation):
	'''Opens either an Archive or a MultiFileArchive, depending on whether 'paths' is a string or a list.'''
	if isinstance(paths, str):
		return Archive(paths, validation)
	return MultiFileArchive(paths, validation)
	
		
def _parallel_epoch_fn(process_index, archive_options, task_queue_data, result_queue_data, quit_flag):
	'''Main function for worker processes that produce randomly augmented minibatches.'''
	archive = open_archive(*archive_options)
	batch_divider = BatchDivider(archive)
	
	task_queue = shared.CircularBuffer(from_shared=task_queue_data)
	result_queue = shared.CircularBuffer(from_shared=result_queue_data)
	
	while not quit_flag.value:
		# Read a task.
		task_queue.lock.acquire()
		
		task = task_queue.consume()
		if task is None:
			# None available right now.
			task_queue.lock.release()
			time.sleep(config.WORKER_DELAY)
			continue
		
		# Got a task!
		task = copy.copy(task)
		task_queue.lock.release()
		
		# Get the crops.
		batch_data = batch_divider.constructBatch(task)
		
		# Add to the cache.
		while not quit_flag.value:
			result_queue.lock.acquire()
			success = result_queue.append(batch_data)
			result_queue.lock.release()
			
			if success:
				break
			else:
				time.sleep(config.WORKER_DELAY)

WAIT_DELAY = 0.00001

LOCK_MAX_TRY_COUNT = 10 # How many times to try non-blocking locking before acquiring the lock forcefully.
def foreach_lock(locks, items, function, run_count=None, retry=True):
	'''Runs the given function for a range of items, locking the corresponding lock
	   before calling the function.
	   
	   Stops after 'run_count' iterations (or earlier if out of locks).
	   If 'retry' is False, tries each lock only once.
	'''
	remaining = len(locks)
	if run_count is not None:
		remaining = min(remaining, run_count)
	
	handled = [False] * len(locks)
	
	try_count = 0
	while remaining:
		for i, item in enumerate(items):
			if handled[i]:
				continue
			
			if locks[i].acquire(block=(try_count >= LOCK_MAX_TRY_COUNT)):
				function(*item)
				
				handled[i] = True
				remaining -= 1
				locks[i].release()
			
				if remaining <= 0:
					break

		if not retry:
			break
			
		try_count += 1
		time.sleep(WAIT_DELAY)

		

def preinit_parallel_epoch(archive, tuple_size=None, mode='random_order'):
	'''Returns a generator that caches minibatches from an archive by using parallel processes
	   and yields complete minibatches.
	   
	   If tuple_size is None, the generator returns a single minibatch each time.
	   Otherwise yields tuples of size 'tuple_size'.
	   
	   Give minibatches=-1 to the generator's constructor for infinite repeating, otherwise specify the number of repeats.
	   
	   Possible modes include:
	       'random_order': Traverses the examples in the dataset in random order.
		   'reverse_digits': Minibatches are stratified by reversing the digits of the index. The selected minibatch depends on timing and is not fully deterministic.
	   
	   Warning: Do not break out from a parallel_epoch or the parallel processes will not be shut down:
	   The generator will shut down the parallel processes when the iteration has finished.
	'''
	
	real_tuple_size = tuple_size if tuple_size is not None else 1
	assert config.WORKER_PROCESS_COUNT > real_tuple_size

	
	# Launch workers.
	quit_flag = multiprocessing.Value('b', False)

	processes = []
	task_queues = []
	result_queues = []
	
	result_dimensions = [config.TRAIN_CACHED_BATCH_COUNT, config.BATCH_SIZE, archive.dimensions, config.TOTAL_SIZE, config.TOTAL_SIZE]
	print("Result size: {} MB".format(result_dimensions[0]*result_dimensions[1]*result_dimensions[2]*result_dimensions[3]*result_dimensions[4]*4/1024/1024))
	result_dtype = np.float16
	
	for i in range(config.WORKER_PROCESS_COUNT):
		task_queue = shared.CircularBuffer(dtype=np.uint32, dimensions=[config.TRAIN_CACHED_BATCH_COUNT, config.BATCH_SIZE])
		result_queue = shared.CircularBuffer(dtype=result_dtype, dimensions=result_dimensions)

		gc.collect() # Does this help with memory use? (multiprocessing.Process forks)
		process = multiprocessing.Process(
			target=_parallel_epoch_fn,
			args=(i, (archive.path, archive.validation), task_queue.getSharedMemory(), result_queue.getSharedMemory(), quit_flag)
		)
		process.start()
		
		
		processes.append(process)
		task_queues.append(task_queue)
		result_queues.append(result_queue)

	# Create the BatchDivider.
	if mode == 'random_order':
		batch_divider = BatchDivider(archive)
	elif mode == 'reverse_digits':
		batch_divider = DigitReversingBatchDivider(archive)
	else:
		raise Exception('Unknown mode for preinit_parallel_epoch')
		
	# Create the generator.
	def run(minibatches=-1):
		if minibatches == -1:
			minibatches = sys.maxsize
		
		# Cached batches that have not been given to anyone yet.
		cached_batches = []

		# Run for a given number of minibatches.
		tasks_remaining = minibatches

		# Feed in and out.
		while tasks_remaining > 0:
			# Add more tasks to the workers if needed.
			
			# Get sizes of task queues.
			free_counts = [0] * len(processes)
			def get_free_counts(i, task_queue):
				free_counts[i] = task_queue.getFreeSize()
			foreach_lock([t.lock for t in task_queues], enumerate(task_queues), get_free_counts, retry=False)

			# Feed more tasks to the workers.
			if tasks_remaining:
				# Create tasks.
				total_required_batch_count = sum(free_counts[i] for i in range(len(processes)))
				while len(cached_batches) < total_required_batch_count:
					cached_batches.append(np.asarray(batch_divider.produceBatchIndices(config.BATCH_SIZE), dtype=np.uint32))
				
				# Assign them to be sent to the processes.
				tasks = [[] for i in range(len(processes))]
				for i in range(len(processes)):
					for j in range(free_counts[i]):
						tasks[i].append(cached_batches.pop(0))

				# Send tasks to workers.
				successful_adds = set()
				def add_tasks(i, task_queue):
					successful_adds.add(i)

					for task in tasks[i]:
						success = task_queue.append(task)
						if not success:
							print("Failed to add a task to task_queue!") # Should never happen.
							pdb.set_trace()
						
				foreach_lock([task_queues[i].lock for i in range(len(task_queues)) if tasks[i]], [(i, task_queues[i]) for i in range(len(task_queues)) if tasks[i]], add_tasks, retry=False)
				
				# Put unsent tasks back to the cache.
				unused_tasks = []
				for i, task_queue in enumerate(task_queues):
					if i not in successful_adds:
						unused_tasks += tasks[i]

				cached_batches = unused_tasks + cached_batches
				
			# Read result counts.
			result_counts = [0] * len(processes)
			def get_result_counts(i, result_queue):
				result_counts[i] = result_queue.getDataSize()
			foreach_lock([r.lock for r in result_queues], enumerate(result_queues), get_result_counts, retry=False)
			
			# Return results.
			return_count = min(real_tuple_size, sum(result_counts))
			if return_count == real_tuple_size:
				# Lock queues starting from the one with the most items ready.
				visiting_order = [i for i in range(len(processes)) if result_counts[i] > 0]
				visiting_order = sorted(visiting_order, key=lambda i: result_counts[i], reverse=True)
				
				# Try to lock enough workers without stalling.
				locked_indices = []
				locked_result_count = 0

				for i in visiting_order:
					if result_queues[i].lock.acquire(block=False):
						# Got one!
						locked_indices.append(i)
						locked_result_count += result_counts[i]
				
				if locked_result_count >= real_tuple_size:
					# Succeeded in locking enough queues!
					results = []
					for i in locked_indices:
						for j in range(result_counts[i]):
							if len(results) < real_tuple_size:
								results.append(result_queues[i].consume())

					# Yield, and keep the workers locked to avoid copying the memory.
					if tuple_size is None:
						tasks_remaining -= 1
						yield results[0]
					else:
						tasks_remaining -= len(results)
						yield results

				# Unlock everything.
				for i in locked_indices:
					result_queues[i].lock.release()
			else:
				time.sleep(config.MAIN_THREAD_DELAY)
			
		# Quit workers.
		quit_flag.value = True
			
		for process in processes:
			process.join()

	return run
