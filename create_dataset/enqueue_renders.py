import math
import random

import numpy as np

import scene_parameters


### Low-discrepancy sequences for rendering crops.

_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
def _g(i, base):
	multiplier = 1.0 / base
	x = 0.0

	while i > 0:
		x += multiplier * (i % base)
		i //= base
		multiplier /= base
	
	return x

def _rotate(data):
	np.random.seed(random.randint(0, 2**32 - 1))
	
	rotation = np.random.uniform(size=[1, data.shape[1]])
	return np.mod(data + rotation, 1.0)
	
def hammersley(dimensions, sample_count):
	data = np.zeros([dimensions, sample_count], dtype=np.float32)
	
	for sample_index in range(sample_count):
		for dimension_index in range(dimensions - 1):
			data[dimension_index, sample_index] = _g(sample_index, _primes[dimension_index])
		
		data[dimensions - 1, sample_index] = float(sample_index) / sample_count
		
	return _rotate(data)


### Output directory names.

def getRealTaskName(scene, image_index, spp_index, prefix=''):
	return '{}-{}-{}-{}'.format(prefix, scene, image_index, spp_index)

def getCleanTaskName(scene, image_index, prefix=''):
	return '{}-{}-{}-{}'.format(prefix, scene, image_index, 'clean')

	
### Scene options.

def sceneOptions(scene_name, image_width, image_height, animation_length, xml_sequence=''):
	return (scene_name, image_width, image_height, animation_length, xml_sequence)
	

### Adding render commands to the batches.

def queueDataGather(gen, scene_options, frames, crops, spps, reference_spp, shutter_times, render_parameters={}, 
			        per_frame_render_parameters=None, output_noisy=True, output_clean=True, prefix=''):
	scene_name, image_width, image_height, animation_length, xml_sequence = scene_options

	assert 'sampler' in render_parameters and render_parameters['sampler'] != 'deterministic'

	if output_noisy:
		count = frames.shape[0]
		
		for image_index in range(count):
			for spp_index in range(spps.shape[0]):
				current_render_parameters = render_parameters.copy()
				current_render_parameters.update({
					'xoffset' : crops[image_index][0],
					'yoffset' : crops[image_index][1],
					'xsize' : crops[image_index][2],
					'ysize' : crops[image_index][3],
					'full_xsize' : crops[image_index][4],
					'full_ysize' : crops[image_index][5]
				})
				
				if per_frame_render_parameters:
					current_render_parameters.update(per_frame_render_parameters[image_index])
			
				# Add the actual render command to the batch configs.
				gen.queueGPT(
					scene = scene_name,
					seed = random.randint(1, 65535),
					spp = spps[spp_index, image_index].item(),
					frame_time = 1.0,
					frame = frames[image_index].item(),
					frame_count = 1,
					shutter_time = shutter_times[image_index].item(),
					task_name = getRealTaskName(scene_name, image_index, spp_index, prefix=prefix),
					xml_sequence = xml_sequence,
					render_parameters = current_render_parameters,
				)

	if output_clean:
		count = frames.shape[0]
		
		for image_index in range(count):
			current_render_parameters = render_parameters.copy()
			current_render_parameters.update({
				'xoffset' : crops[image_index][0],
				'yoffset' : crops[image_index][1],
				'xsize' : crops[image_index][2],
				'ysize' : crops[image_index][3],
				'full_xsize' : crops[image_index][4],
				'full_ysize' : crops[image_index][5]
			})
			
			if per_frame_render_parameters:
				current_render_parameters.update(per_frame_render_parameters[image_index])
			
			# Add the actual render command to the batch configs.
			gen.queueGPT(
				scene = scene_name,
				seed = 0, 
				spp = reference_spp,
				frame_time = 1.0,
				frame = frames[image_index].item(),
				frame_count = 1,
				shutter_time = shutter_times[image_index].item(),
				task_name = getCleanTaskName(scene_name, image_index, prefix=prefix),
				xml_sequence = xml_sequence,
				render_parameters = current_render_parameters
			)


### Getting random configuration parameters from the LDS.

def getRandomSpps(min_spp, max_spp, images_per_crop, random_1d):
	log_max = np.log(max_spp)
	log_min = np.log(min_spp)
	
	vectors = []
	for i in range(images_per_crop):
		vectors.append(np.mod(random_1d + float(i) / float(images_per_crop), 1.0))
	
	random_nd = np.stack(vectors, axis=0)
	
	log_random = log_min + (log_max - log_min) * random_nd
	return np.exp(log_random).astype(np.int32)
	
def getShutterTimes(min_shutter, max_shutter, random_1d):
	if min_shutter == max_shutter:
		return np.ones_like(random_1d) * min_shutter
	else:
		return min_shutter + (max_shutter - min_shutter) * random_1d
	
def getFrames(scene_options, random_1d):
	scene_name, image_width, image_height, animation_length, xml_sequence = scene_options
	
	frames = (random_1d * animation_length).astype(np.int32)
	return frames
		
def getCrops(scene_options, crop_width, crop_height, random_2d):
	scene_name, image_width, image_height, animation_length, xml_sequence = scene_options

	crop_count = random_2d.shape[1]
	
	crops = []
	for i in range(crop_count):
		crops.append((
			int(random_2d[0][i] * (image_width - crop_width)),
			int(random_2d[1][i] * (image_height - crop_height)),
			crop_width,
			crop_height,
			image_width,
			image_height
		))

	return crops
	
def getScaledCrops(scene_options, crop_width, crop_height, random_3d):
	scene_name, image_width, image_height, animation_length, xml_sequence = scene_options

	width_scale = float(crop_width) / image_width
	height_scale = float(crop_height) / image_height	
	assert width_scale <= 1 and height_scale <= 1
	
	min_scale = max(width_scale, height_scale)
	max_scale = 1.0
	
	crop_count = random_3d.shape[1]
	
	crops = []
	for i in range(crop_count):
		scale = min_scale + (max_scale - min_scale) * random_3d[2][i]
		full_crop_width = int(math.ceil(scale * image_width))
		full_crop_height = int(math.ceil(scale * image_height))

		crops.append((
			int(random_3d[0][i] * (full_crop_width - crop_width)),
			int(random_3d[1][i] * (full_crop_height - crop_height)),
			crop_width,
			crop_height,
			full_crop_width,
			full_crop_height
		))

	return crops


def render_training_crops(gen, scene_name, crop_size, shutter_range, crop_count, images_per_crop, spp_range, reference_spp, prefix, animation_length=120, xml_sequence='', output='noisy', seed=None):
	assert output in ('noisy', 'clean')
	assert seed is not None

	print("Writing batch for '{}'... ".format(scene_name))
	
	scene_options = sceneOptions(scene_name, 1280, 720, animation_length, xml_sequence=xml_sequence) 
	
	random.seed(seed)	
	
	samples = hammersley(7, crop_count)
	
	# Rendering parameters.
	render_parameters = {
		'sampler': 'independent' if output == 'noisy' else 'ldsampler',
		'recfilter': 'box'
	}
	
	per_frame_render_parameters = []	
	for i in range(crop_count):
		beginTarget, endTarget, beginOrigin, endOrigin, beginUp, endUp, focusDist, apertureRad = scene_parameters.SampleSensorParameters(scene_name, useTargets=True, useDof=True)
		
		per_frame_render_parameters.append({
			'beginAnimation': '0.0',
			'endAnimation': '1.0',
			'focusDistance': focusDist,
			'apertureRadius': apertureRad,
			'beginOrigin': beginOrigin,		
			'endOrigin': endOrigin,
			'beginTarget': beginTarget,
			'endTarget': endTarget,
			'beginUp': beginUp,
			'endUp': endUp,
		})
	
	# Support for multi-xml scenes.
	if xml_sequence:
		frames = getFrames(scene_options, samples[0, :])
	else:
		frames = np.zeros_like(getFrames(scene_options, samples[0, :]))
	
	# Add the renders to the batch files.
	queueDataGather(
		gen,
		scene_options,
		frames,
		getScaledCrops(scene_options, crop_size, crop_size, samples[1:4, :]),
		getRandomSpps(spp_range[0], spp_range[1], images_per_crop, samples[5, :]),
		reference_spp,
		getShutterTimes(shutter_range[0], shutter_range[1], samples[6, :]),
		render_parameters,
		per_frame_render_parameters = per_frame_render_parameters,
		output_noisy=(output == 'noisy'),
		output_clean=(output == 'clean'),
		prefix=prefix
	)

	print("Done.")

def render_test_images(gen, scene_name, shutter_range, crop_count, images_per_crop, reference_spp, prefix, animation_length=120, xml_sequence='', output='noisy', use_pt=False, seed=None):
	assert output in ('noisy', 'clean')
	assert seed is not None

	print("Writing batch for '{}'... ".format(scene_name))
	
	scene_options = sceneOptions(scene_name, 1280, 720, animation_length, xml_sequence=xml_sequence)
	
	random.seed(seed)
	
	samples = hammersley(3, crop_count)
	
	# Rendering parameters.
	render_parameters = {
		'sampler': 'independent' if output == 'noisy' else 'ldsampler',
		'recfilter': 'box'
	}
	
	if use_pt:
		render_parameters['disableGradients'] = True
		base_spp = 2.5
	else:
		base_spp = 1.0
	
	spps = np.tile(0.5 + base_spp * (2 ** np.linspace(0, images_per_crop - 1, num=images_per_crop).reshape([-1, 1])), [1, crop_count]).astype(np.int32)
	frames = np.linspace(0, animation_length - 1, num=crop_count).astype(np.int32)
	print(frames)
	
	queueDataGather(
		gen,
		scene_options,
		frames,
		getCrops(scene_options, 1280, 720, samples[0:2, :]), # Note: samples not used!
		spps,
		reference_spp,
		getShutterTimes(shutter_range[0], shutter_range[1], samples[2, :]),
		render_parameters=render_parameters, 
		output_noisy=(output == 'noisy'),
		output_clean=(output == 'clean'),
		prefix=prefix
	)
	
	print("Done.")
