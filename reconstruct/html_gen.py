import string
import os
import pdb
import io
import base64
import struct
import multiprocessing
import time
import copy
import config

import imageio

import shared
import numpy as np


# Size of the input circular buffer for the workers for task allocation
TASK_QUEUE_LENGTH = 5 * config.PNG_WORKER_PROCESS_COUNT

# Internal batch size for parallel processing. How many images are compressed in one batch before repeating.
MAXIMUM_TASK_SIZE = 100


HEADER = string.Template('''
<!doctype html>
<head><title>$Title</title></head>
<article>
<style type="text/css">
$CSS
</style>
<h1>$Title</h1>
''')

FOOTER = string.Template('''
<script>
$Script
</script>
</article>
</html>''')

CSS = '''
.image_comparison_example {
  background-color: #DDD;
  overflow:auto;
}

.image_comparison_example:nth-child(even) {
  background-color: #FFF;
  overflow:auto;
}

.image_comparison_group {
  display: inline-block;
  vertical-align: middle;
  margin: 1px;
  background-color: #CCF;
  border: 1px solid;
}
.image_comparison_group:hover {
  border-color: yellow;
}
.image_comparison_group:active {
  border-color: red;
}

.image_comparison_item {
  display: block;
  width: 96px;
  height: 96px;
}

.image_comparison_example_name {
  vertical-align: middle;
  padding: 0;
  margin: 0;
}

.image_comparison_group_name {
  text-align: center;
  padding: 0;
  margin: 0;
}

.image_comparison_canvas {
  background-color: #FFA;
  vertical-align: middle;
  border-style: solid;
  border-width: 1px;
  margin-right: 1px;
}

.image_comparison_menu {
  display: inline-flex;
  flex-wrap: wrap;
}

.image_comparison_menu_item {
  margin: 0;
  padding: 0;
  border: 1px solid;
  width: 128px;
  height: 128px;
}
.image_comparison_menu_item:hover {
  border-color: yellow;
}
.image_comparison_menu_item:active {
  border-color: red;
}'''

SCRIPT = string.Template('''
var image_comparisons = {};
var image_cache = {};

var images_base64 = {};

function loadImage(path, onload) {
	if((path) in image_cache) {
		var image = image_cache[path];
		onload();
	} else {
		var image = new Image();
		
		image.onload = function() {
			image_cache[path] = image;
			onload(image);
		}
		
		image.src = images_base64[path];
	}
}

function loadImages(paths, onload) {
	var images_remaining = paths.length;


	var single_onload = function(image) {
		--images_remaining;
		if(images_remaining == 0) {
			var images = [];
			for(var i = 0; i < paths.length; ++i) {
				images.push(image_cache[paths[i]]);
			}
			
			onload(images);
		}
	};
	
	for(var i = 0; i < paths.length; ++i) {
		loadImage(paths[i], single_onload);
	}
}

function createImageComparison(comparison_name, example_count, group_names, default_group, image_type_count) {
	image_comparison = {
		name:comparison_name,
		active_index:0,
		active_group:default_group,
		example_count:example_count,
		group_names:group_names,
		image_type_count:image_type_count
	};
	
	image_comparisons[comparison_name] = image_comparison;
	
	updateImageComparisonMenu(comparison_name);
	updateImageGroups(comparison_name);
	updateImageComparison(comparison_name);
	
	return image_comparison;
}
	
function getImageComparison(comparison_name) {
	return image_comparisons[comparison_name];
}

function setActiveExample(comparison_name, example_index) {
	var image_comparison = image_comparisons[comparison_name];
	image_comparison.active_index = example_index;

	updateImageComparison(comparison_name);
	updateImageGroups(comparison_name);
}

function setActiveGroup(comparison_name, group_name) {
	var image_comparison = image_comparisons[comparison_name];
	image_comparison.active_group = group_name;

 	updateImageComparison(comparison_name);
	updateImageComparisonMenu(comparison_name);
}


function updateImageComparisonMenu(comparison_name) {
	var image_comparison = getImageComparison(comparison_name);
	var example_index = image_comparison.active_index;
	var example_count = image_comparison.example_count;
	var group_name = image_comparison.active_group;
	for(var i = 0; i < example_count; ++i) {
		var img = document.getElementById(comparison_name + '-menu-input-' + i);
		img.src = images_base64[comparison_name + "-" + group_name + "-" + i + "-0.png"];
	}
}

function updateImageGroups(comparison_name) {
	var image_comparison = getImageComparison(comparison_name);
	var example_index = image_comparison.active_index;
	var group_name = image_comparison.active_group;
	var image_type_count = image_comparison.image_type_count;
	var group_names = image_comparison.group_names;
	
	for(var group_index = 0; group_index < group_names.length; ++group_index) {
		for(var image_index = 0; image_index < image_type_count; ++image_index) {
			var img = document.getElementById(comparison_name + '-' + group_names[group_index] + '-' + image_index);
			img.src = images_base64[comparison_name + '-' + group_names[group_index] + '-' + example_index + '-' + image_index + ".png"];
		}
	}
}

function updateImageComparison(comparison_name) {
	var image_comparison = getImageComparison(comparison_name);
	var example_index = image_comparison.active_index;
	var group_name = image_comparison.active_group;
	var image_type_count = image_comparison.image_type_count;
	
	var paths = [];
	
	for(var i = 0; i < image_type_count; ++i) {
		paths.push(comparison_name + "-" + group_name + "-" + example_index + "-" + i + ".png");
	}
	for(var i = 0; i < image_type_count; ++i) {
		paths.push(comparison_name + "-" + 'reference' + "-" + example_index + "-" + i + ".png");
	}
	
	loadImages(paths, function(images) {
		var canvas = document.getElementById("canvas_" + comparison_name)
		var ctx = canvas.getContext("2d");
		
		var canvas_width = canvas.width;
		var canvas_height = canvas.height;

		for(var i = 0; i < image_type_count; ++i) {
			let j = i + image_type_count;
			ctx.drawImage(images[i], i * images[0].width, 1 * images[0].height,  images[0].width, images[0].height);
			ctx.drawImage(images[j], i * images[0].width, 0 * images[0].height, images[0].width, images[0].height);
		}
		
		var source = ctx.getImageData(0, images[0].height, canvas_width, images[0].height);
		var reference = ctx.getImageData(0, 0, canvas_width, images[0].height);
		var target = ctx.getImageData(0, 2 * images[0].height, canvas_width, images[0].height);
		
		var src_data = source.data;
		var ref_data = reference.data;
		var tar_data = target.data;
				
		var pow_lookup = [];
		for(var i = 0; i < 256; ++i) {
			pow_lookup.push(Math.pow((i / 255.0), 2.2))
		}
		for(var i = 0; i < canvas_height; ++i) {
			var offset = i * canvas_width * 4;
			
			for(var j = 0; j < canvas_width; ++j) {
				var pos_err = 0;
				var neg_err = 0;
				
				for(var k = 0; k < 3; ++k) {
					var src_value = pow_lookup[src_data[offset]];
					var ref_value = pow_lookup[ref_data[offset]]
					pos_err = Math.max(pos_err, src_value - ref_value);
					neg_err = Math.min(neg_err, src_value - ref_value);
					++offset;
				}
				++offset;

				tar_data[offset-4] = 255.0 * Math.pow(Math.max(0, pos_err), 1.0/2.2);
				tar_data[offset-3] = 255.0 * Math.pow(Math.max(0, - neg_err), 1.0/2.2);
				tar_data[offset-2] = 0;
				tar_data[offset-1] = 255;
			}
		}
		ctx.putImageData(target, 0,  2 * images[0].height);
	});
}


$CustomScript
''')


IMAGE_COMPARISON_BEGIN = string.Template('''
<h1>$ComparisonName<span id="${ComparisonName}_loadbar"></span></h1>

'''
)
IMAGE_COMPARISON_END = string.Template('''
''')

IMAGE_COMPARISON_MENU_BEGIN = string.Template('''
<div class="image_comparison_menu"><!--'''
)
IMAGE_COMPARISON_MENU_ITEM = string.Template('''
	--><img src="" class="image_comparison_menu_item" id="$ComparisonName-menu-input-$ExampleIndex"  onclick="setActiveExample('$ComparisonName', $ExampleIndex)"><!--''')
IMAGE_COMPARISON_MENU_END = string.Template('''
--></div>''')

IMAGE_COMPARISON_EXAMPLE_BEGIN = string.Template('''
<div class="image_comparison_example">
	<canvas class="image_comparison_canvas" id="canvas_${ComparisonName}" width="$CanvasWidth" height="$CanvasHeight" style="width:${CanvasZoomWidth}px;height:${CanvasZoomHeight}px"></canvas><!--'''
)
IMAGE_COMPARISON_EXAMPLE_END = string.Template('''
--></div>'''
)

IMAGE_COMPARISON_ROW_BEGIN = string.Template('''
--><div class="image_comparison_group" id="$ComparisonName-$GroupName" onmouseover="setActiveGroup('$ComparisonName', '$GroupName')">
	<p class="image_comparison_group_name">$GroupName</p>'''
)
IMAGE_COMPARISON_ROW_END = string.Template('''
	</div><!--'''
)

IMAGE_COMPARISON_ROW_ITEM = string.Template('''
		<img src="" class="image_comparison_item" id="$ComparisonName-$GroupName-$ItemNumber">'''
)

IMAGE_COMPARISON_SCRIPT = string.Template('''
createImageComparison('$ComparisonName', $ExampleCount, [$GroupNameList], '$DefaultGroup', $GroupSize);'''
)


INFO = string.Template('''
<p>$Message</p>''')


def _parallel_png_fn(task_def, image_def, output_def, quit_flag):
	task_dimensions, task_shared = task_def
	image_dimensions, image_shared = image_def
	output_count, output_shared = output_def
	
	tasks = shared.CircularBuffer(dtype=np.uint32, dimensions=task_dimensions, from_shared=task_shared)
	images = shared.CircularBuffer(dtype=np.uint8, dimensions=image_dimensions, from_shared=image_shared)
	output = shared.BytesArray(from_shared=output_shared)
	
	while not quit_flag.value:
		# Get a task.		
		tasks.lock.acquire()

		task = tasks.consume()
		if task is None:
			tasks.lock.release()
			time.sleep(config.PNG_WORKER_DELAY)
			continue
		
		image = images.consume()
		
		# Got the task!
		task = copy.copy(task)
		image = copy.copy(image)
		
		tasks.lock.release()
		
		# Compress.
		memory_file = io.BytesIO()
		imageio.imwrite(memory_file, image, format='.png')
		png_bytes = memory_file.getvalue()
		
		# Add to output.
		output.lock.acquire()
		output[task] = png_bytes
		output_count.value += 1
		output.lock.release()

class ParallelCompressor:
	def __init__(self, worker_count, maximum_task_size, dimensions):
		'''Parameters:
			worker_count: number of worker processes
			maximum_task_size: size of the internal parallel batches that larger compression jobs are split into
			dimensions: image dimensions (height, width, channels)'''
			
		# Create shared memory.
		self.task_dimensions = [TASK_QUEUE_LENGTH]
		self.image_dimensions = [TASK_QUEUE_LENGTH, dimensions[0], dimensions[1], dimensions[2]]
		self.maximum_task_size = maximum_task_size
		
		self.tasks = shared.CircularBuffer(dtype=np.uint32, dimensions=self.task_dimensions)
		self.images = shared.CircularBuffer(dtype=np.uint8, dimensions=self.image_dimensions)
		
		# Buffer for the compressed output data. Reserve 1.2 the size of the uncompressed images just to be sure.
		self.output = shared.BytesArray(dimensions[0] * dimensions[1] * dimensions[2] * maximum_task_size * 12 // 10) 
		self.output_count = multiprocessing.Value('i', 0)
		
		self.quit_flag = multiprocessing.Value('b', False)
		
		self.task_def = self.task_dimensions, self.tasks.getSharedMemory()
		self.image_def = self.image_dimensions, self.images.getSharedMemory()
		self.output_def = self.output_count, self.output.getSharedMemory(),

		# Launch workers.	
		self.processes = []
		for i in range(worker_count):
			process = multiprocessing.Process(
				target=_parallel_png_fn,
				args=(self.task_def, self.image_def, self.output_def, self.quit_flag)
			)
			process.start()
			self.processes.append(process)
	
		self.quitted = False
	
	def quit(self):
		if not self.quitted:
			self.quitted = True
			self.quit_flag.value = True
			for process in self.processes:
				process.join()
		
	def __del__(self):
		self.quit()
	
	def compress(self, data):
		results = []

		batch_begin = 0
		while batch_begin < data.shape[0]:
			batch_size = min(data.shape[0] - batch_begin, self.maximum_task_size)
			tasks_remaining = list(range(batch_begin, batch_begin + batch_size))
						
			self.output_count.value = 0
			self.output.clear()

			# Feed in and out.
			while True:
				# See if we're ready.
				self.output.lock.acquire()
				if self.output_count.value == batch_size:
					self.output.lock.release()
					break
				self.output.lock.release()

				# Add more tasks to the workers if needed.
				self.tasks.lock.acquire()
				current_task_queue_length = self.tasks.getDataSize()
				self.tasks.lock.release()
				
				need_more = min(TASK_QUEUE_LENGTH - current_task_queue_length, len(tasks_remaining))
				if need_more > 0:
					self.tasks.lock.acquire()

					for new_task_index in range(need_more):
						self.tasks.append(tasks_remaining[0] - batch_begin)
						self.images.append(data[tasks_remaining[0]])
						tasks_remaining.pop(0)
						
					self.tasks.lock.release()
				else:
					time.sleep(config.PNG_MAIN_THREAD_DELAY)
						
			# Collect results.
			for i in range(batch_size):
				results.append(self.output[i])
		
			batch_begin += batch_size

		return results


class Info:
	def __init__(self, name):
		self.name = name
		self.message = ''
	
	def setMessage(self, message):
		self.message = message
	
	def compile(self):
		markup = []
		
		markup.append(INFO.substitute({
			'Message': self.message
		}))
		
		return "\n".join(markup), ""

def getCachedCompressor(compressors, height, width, channels):
	# Get or cache a compressor.
	key = ('png', height, width, channels)
	dimensions = [height, width, channels]
	
	if not key in compressors:
		print("Creating workers for PNG encoding.")

		compressor = ParallelCompressor(config.PNG_WORKER_PROCESS_COUNT, MAXIMUM_TASK_SIZE, dimensions)
		compressors[key] = compressor
	
	return compressors[key]
	

class ImageComparison:
	def __init__(self, name, directory, default_group, compressors):
		self.name = name
		self.groups = [] # Note: Methods.
		self.directory = directory
		self.default_group = default_group
		self.compressors = compressors
		
		self.static_cache = {}
		self.compressor = None
	
	def clear(self):
		self.groups = []
		
	def add(self, group_name, group_images, static=False):
		if group_images[0].shape[3] == 0:
			pdb.set_trace() # Erroneous number of channels.
		
		
		self.groups.append((group_name, group_images, static))
	
	def compile(self):
		markup = []
		script = []
		
		# Get image metadata.
		example_count = self.groups[0][1][0].shape[0]
		example_width, example_height = self.groups[0][1][0].shape[2], self.groups[0][1][0].shape[1]
		group_size = len(self.groups[0][1]) # Primal, dx, dy, etc.
		
		current_index = 0
		current_percentage = -1
		
		# Collect images that need to be compressed.
		compressable_images_index = {}
		compressable_images = []
		for group_name, group_images, group_is_static in self.groups:
			for image_type_index, image_batch in enumerate(group_images):
				for example_index in range(example_count):
					key = (group_name, example_index, image_type_index)
					if key not in self.static_cache:
						compressable_images_index[(group_name, example_index, image_type_index)] = len(compressable_images)
						if example_index >= image_batch.shape[0]:
							raise Exception('Group "{}" has too few images.'.format(group_name))
						
						compressable_images.append(image_batch[example_index, :, :, :])
					
		
		compressable_images = np.stack(compressable_images)
		
		# Get a compressor.
		if self.compressor is None:
			self.compressor = getCachedCompressor(self.compressors, compressable_images.shape[1], compressable_images.shape[2], compressable_images.shape[3])
		compressor = self.compressor
			
		compressed_pngs = compressor.compress(compressable_images)
		
		# Add to the page.
		for group_name, group_images, group_is_static in self.groups:
			for image_type_index, image_batch in enumerate(group_images):
				for example_index in range(example_count):
					path = "{}-{}-{}-{}.png".format(self.name, group_name, example_index, image_type_index)
					
					# Get compressed data.
					key = (group_name, example_index, image_type_index)
					if key in self.static_cache:
						png = self.static_cache[key]
					else:
						png = compressed_pngs[compressable_images_index[key]]
						if group_is_static:
							self.static_cache[key] = png
					
					image_base64 = "data:image/png;base64," + base64.b64encode(png).decode('ascii')
					script.append("images_base64['" + path + "'] = '" + image_base64 + "';")
					
					# Update loadbar.
					current_index += 1
					percentage = (100 * current_index) // (len(self.groups) * len(group_images) * example_count)
					if percentage > current_percentage:
						current_percentage = percentage
						script.append("document.getElementById('{}_loadbar').innerHTML='({}%...)';</script><script>".format(self.name, percentage))
						
		script.append("document.getElementById('{}_loadbar').innerHTML='';</script><script>".format(self.name))

		# Title			
		markup.append(IMAGE_COMPARISON_BEGIN.substitute({'ComparisonName': self.name}))
		
		# Menu
		markup.append(IMAGE_COMPARISON_MENU_BEGIN.substitute({}))
		for example_index in range(example_count):
			markup.append(IMAGE_COMPARISON_MENU_ITEM.substitute({
				'ComparisonName': self.name,
				'ExampleIndex': str(example_index)
			}))
		markup.append(IMAGE_COMPARISON_MENU_END.substitute({}))
		
		# View
		
		#for example_index in range(example_count):
		markup.append(IMAGE_COMPARISON_EXAMPLE_BEGIN.substitute({
			'ComparisonName': self.name,
			'CanvasWidth': str(group_size * example_width),
			'CanvasHeight': str(3 * example_height),
			'CanvasZoomWidth': str(group_size * 256), #4 * group_size * example_width),
			'CanvasZoomHeight': str(3 * 256),
		}))

		for item_index, (group_name, group_images, group_is_static) in enumerate(self.groups):
			markup.append(IMAGE_COMPARISON_ROW_BEGIN.substitute({
				'ComparisonName': self.name,
				'GroupName': group_name,
				'ItemCount': str(len(group_images)),
				'ItemNumber': str(item_index),
			}))
			
			for image_type_index, image_batch in enumerate(group_images):
				markup.append(IMAGE_COMPARISON_ROW_ITEM.substitute({
					'ComparisonName': self.name,
					'ExampleNumber': 0,
					'GroupName': group_name,
					'ItemNumber': str(image_type_index),
				}))
				
			markup.append(IMAGE_COMPARISON_ROW_END.substitute({
			}))
	
		markup.append(IMAGE_COMPARISON_EXAMPLE_END.substitute({}))

		script.append(IMAGE_COMPARISON_SCRIPT.substitute({
			'ComparisonName': self.name,
			'ExampleCount': str(example_count),
			'DefaultGroup': self.default_group,
			'GroupNameList': ", ".join(["'{}'".format(group_name) for group_name, _, _ in self.groups]),
			'GroupSize': str(group_size)
		}))
		
		markup.append(IMAGE_COMPARISON_END.substitute({}))
		
		return '\n'.join(markup), '\n'.join(script)
		

class HtmlGenerator:
	def __init__(self, directory, title=''):
		self.directory = directory

		self.title = title
		
		self.page = []
		self.contents = {}
		
		self.compressors = {}
		
		self.quitted = False
	
	def quit(self):
		if not self.quitted:
			self.quitted = True
			for compressor in self.compressors.values():
				compressor.quit()
			
	def __del__(self):
		self.quit()
		
	def setTitle(self, title):
		self.title = title
		
	def info(self, name):
		item = Info(name)
		if not name in self.contents:
			self.page.append(name)
		
		self.contents[name] = item
		return item

	def precreateCompressor(self, height, width, channels):
		getCachedCompressor(self.compressors, height, width, channels)
	
	def imageComparison(self, name, default_group):
		if name in self.contents:
			item = self.contents[name]
			assert isinstance(item, ImageComparison)
			
			item.clear()
			return item
			
		item = ImageComparison(name, self.directory, default_group, self.compressors)
		if not name in self.contents:
			self.page.append(name)
		
		self.contents[name] = item
		return item
		
	def write(self):
		os.makedirs(self.directory, exist_ok=True)
		
		markup = []
		custom_script = []
		
		markup.append(HEADER.substitute({
			'Title': self.title,
			'CSS': CSS
		}))

		for item_name in self.page:
			item_markup, item_script = self.contents[item_name].compile()
			markup.append(item_markup)
			custom_script.append(item_script)

		script = SCRIPT.substitute({
			'CustomScript': '\n'.join(custom_script)
		})
		markup.append(FOOTER.substitute({
			'Script': script
		}))
		
		with open(os.path.join(self.directory, 'index.html'), 'w') as file:
			file.write('\n'.join(markup))
	