import os
from util import int_arg


### Model defaults.
class ModelConfig:
	def __init__(self):
		### Pooling options.

		# Type of pooling/downsampling used. Options: 'conv', 'lanczos', 'average', 'max'.
		self.pool_type = 'conv'
				
		# Type on unpooling/upsampling used. Options: 'conv', 'lanczos', 'bilinear', 'nearest'
		self.unpool_type = 'conv'

			
		### Network size.
			
		# Number of levels/resolutions in the network. Supported: 1-5.
		self.levels = 4
			
		# Number of features added by each processing unit.
		self.growth_rates = [40, 80, 80, 80, None]
			
		# Number of processing units before and after pooling.
		self.unit_counts = [4, 3, 2, 2, None]
			
		# Number of features passed to the next half-resolution.
		# The first number must be None as the first layer receives no inputs from a higher resolution.
		self.downscale_feature_counts = [None, 160, 160, 160, None]
			
		# Number of features passed back to the next double resolution.
		self.upscale_feature_counts = [80, 160, 160, None, None]
			
			
		# Amount of leakiness in the leaky ReLU. Use 0.0 for a non-leaky rectifier.
		self.relu_factor = 0.01
			

		### Learning rate options.
			
		# Base learning rate, reached after an initial warm-up period and ramped down as the training progresses.
		self.learning_rate = 5e-4
			
		# How much the learning rate is pressed down initially.
		self.warmup_factor = 1e4
			
		# Number of minibatches over which the learning rate is geometrically raised to full speed.
		self.warmup_length = 10000
					
		# Number of minibatches to run at full learning rate before rampdown calculations start.
		# Set to None to disable rampdown.
		self.rampdown_begin = 50000
			
		# Length of a rampdown stage. The main rampdown variable to be tuned for a new model.
		self.rampdown_stage_length = 70000
			
		# Multiplier to apply to the learning rate after each rampdown stage:
		#     lrate = lrate0 * decay_per_stage^stage.
		self.rampdown_decay_per_stage = 0.5
			
		# Type of rampdown: 'geometric' or 'exponential'.
		#
		# 'geometric' changes the learning rate in discrete steps while 'exponential' changes it continuously.
		# Use geometric for choosing a good stage length, then switch to exponential.
		self.rampdown_func = 'exponential'
		
		# IMPORTANT: Disable the rampdown stage initially when dealing with a new model.
		#
		#            After a new model produces good results, set 'rampdown_func' to 'geometric'
		#            and make sure that the validation loss approximately levels before the next
		#            rampdown stage is reached. The main variable to be tuned is 'rampdown_stage_length'.
		#            Once the learning rate levels approximately in the length of a stage, you
		#            can set 'rampdown_func' back to 'exponential' for a smooth rampdown curve.

			
		### Ensemble loss options.
		
		# Number of times to evaluate the stochastic E-LPIPS loss for each training result.
		#
		# Value 3 is a good default for heavier models and 1 is a good default for small models.
		# Value None disables evaluation of E-LPIPS.
		self.elpips_eval_count = None

		### Enabled features. Primal color is always enabled.
		self.features = {
			'gradients': True,
			'albedo': True,
			'normal': True,
			'depth': True,
			'diffuse': False,
			'specular': False,
			'variances': False
		}
		
		### Enabled augmentations.
		self.augmentations = {
			'flip_x': True,
			'flip_y': True,
			'swap_xy': True,
			'permute_rgb': True,
			'brightness': True,
			'color_multiply': True
		}
		
		# How much padding is varied in both X and Y directions. Must be less than PAD_WIDTH.
		self.vary_padding = 7 #px
		
		
	def input_enabled(self, feature):
		'''Returns whether a given feature input is enabled.'''
		assert feature in {'gradients', 'albedo', 'normal', 'depth', 'variances', 'diffuse', 'specular'}
		return self.features[feature]
		
	def augmentation_enabled(self, augmentation):
		'''Returns whether a gievn feature augmentation is enabled for training.'''
		assert augmentation in {'flip_x', 'flip_y', 'swap_xy', 'permute_rgb', 'brightness', 'color_multiply'}
		return self.augmentations[augmentation]
		
	def validate(self):
		if MODEL_INDEX < 0:
			raise Exception("Model not specified with --config!")
		
		for feature in self.features:
			assert self.input_enabled(feature) in (True, False)
		for augmentation in self.augmentations:
			assert self.augmentation_enabled(augmentation) in (True, False)
		
		assert PAD_WIDTH > 0
		assert self.vary_padding < PAD_WIDTH
		assert self.name
		assert self.loss
		

### Custom model configurations. Choose with --config=N.
def get_model_config():
	model = ModelConfig()
	
	# Losses.
	loss_elpips_squeeze_maxpool = lambda metrics: metrics['elpips_squeeze_maxpool'] + 0.01 * metrics['L1_tonemap']  + 0.01 * metrics['grad_L1_tonemap']
	
	loss_tonemap_l1 = lambda metrics: metrics['L1_tonemap'] + 0.01 * metrics['grad_L1_tonemap']
	
	
	# Model configurations.
	if MODEL_INDEX == 100: # Main configuration.
		model.name = '100_ngpt_elpips_sqz'
		model.elpips_eval_count = 3
		model.loss = loss_elpips_squeeze_maxpool
		
	elif MODEL_INDEX == 101: # Train without gradients.
		model.name = '101_nograd_elpips_sqz'
		model.elpips_eval_count = 3
		model.loss = loss_elpips_squeeze_maxpool
		model.features['gradients'] = False
		
	elif MODEL_INDEX == 102: # Main, but use the L1 loss.
		model.name = '102_ngpt_l1'
		model.loss = loss_tonemap_l1
		model.rampdown_stage_length = 85000.0
		
	elif MODEL_INDEX == 103: # No gradients, L1 loss
		model.name = '103_nograd_l1'
		model.loss = loss_tonemap_l1
		model.rampdown_stage_length = 85000.0
		model.features['gradients'] = False

	elif MODEL_INDEX == 104: # Main, but use Lanczos resampling
		model.name = '104_variant_lanczos'
		model.elpips_eval_count = 3
		model.loss = loss_elpips_squeeze_maxpool
		model.pool_type = 'lanczos'
		model.unpool_type = 'lanczos'
		
	elif MODEL_INDEX == 105: # Main, but use bilinear resampling
		model.name = '105_variant_bilinear'
		model.elpips_eval_count = 3
		model.loss = loss_elpips_squeeze_maxpool
		model.pool_type = 'average'
		model.unpool_type = 'bilinear'

	elif MODEL_INDEX == 106: # Main, but use average pooling
		model.name = '106_variant_avg'
		model.elpips_eval_count = 3
		model.loss = loss_elpips_squeeze_maxpool
		model.pool_type = 'average'
		model.unpool_type = 'nearest'

	elif MODEL_INDEX < 0:
		# No model.
		pass
	else:
		raise Exception('Unknown model index via --config.')

	return model

	
### (Do not modify these.) ###
#
# (Model index given via --config.)
MODEL_INDEX = int_arg('config', default_value=-1)
#
# (Get the used model config.)
model = get_model_config()
#


### Directory configuration.

# Base directory of the project: Parent of config.py.
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# Where the datasets are stored.
DATASET_DIR = BASE_DIR + r'/datasets'

# Directory where to store TensorBoard summaries.
SUMMARY_DIR = BASE_DIR + r'/tensorflow/summaries'

# Where the model parameters are saved.
SAVE_DIRECTORY = BASE_DIR + r'/tensorflow/saves'

# Where to output the HTML visualization files.
HTML_DIR = BASE_DIR + r'/html/result_html_{}'.format(MODEL_INDEX)

# Darc datasets used for training.
TRAIN_DATASET_PATHS = [
#    DATASET_DIR + "/train_default-bathroom-250-5.darc",
    DATASET_DIR + "/train_default-bathroom3-250-5.darc",
    DATASET_DIR + "/train_default-classroom-250-5.darc",
    DATASET_DIR + "/train_default-crytek_sponza-250-5.darc",
#    DATASET_DIR + "/train_default-kitchen_simple-250-5.darc",
    DATASET_DIR + "/train_default-living-room-250-5.darc",
    DATASET_DIR + "/train_default-living-room-2-250-5.darc",
    DATASET_DIR + "/train_default-living-room-3-250-5.darc",
    DATASET_DIR + "/train_default-staircase-250-5.darc",
    DATASET_DIR + "/train_default-staircase2-250-5.darc",
]

# Darc datasets used for validation loss calculation.
VALIDATION_DATASET_PATHS = [
    DATASET_DIR + "/train_default-bathroom2-250-5.darc",
#    DATASET_DIR + "/train_default-bookshelf_rough2-250-5.darc",
    DATASET_DIR + "/train_default-new_kitchen_animation-250-5.darc",
]

# Darc datasets used for the HTML visualization.
VISUALIZATION_DARC_PATHS = VALIDATION_DATASET_PATHS


### Run configuration.


# Size of the training crops.
CROP_SIZE = 128

# Size of minibatch.
BATCH_SIZE = 20

# Number of pixels to add to the training inputs in every direction. Assumed > 0.
PAD_WIDTH = 8
assert PAD_WIDTH > 0

# Number of minibatches to run.
LAST_MINIBATCH = 3000000

# How often to save network weights (minibatch count).
SAVING_PERIOD = 1000 


### Parallelization configuration.

# Number of parallel processes to use for reading the training dataset.
WORKER_PROCESS_COUNT = 6

# Number of parallel processes to use for compressing PNGs for the HTML visualization.
PNG_WORKER_PROCESS_COUNT = 6

# (Less important options)
# Seconds the dataset reader processes wait if their task queue is empty.
WORKER_DELAY = 0.001 
# Seconds the main thread waits if the result queue is empty in dataset reading.
MAIN_THREAD_DELAY = 0.001
# How many minibatches each worker caches.
TRAIN_CACHED_BATCH_COUNT = 10
# Order of traversing the training dataset. Choices: 'random_order', 'reverse_digits'
PARALLEL_EPOCH_MODE = 'random_order'
# Seconds a PNG compression worker waits if out of tasks.
PNG_WORKER_DELAY = 0.05
# Seconds the PNG compressor's main thread waits if the result queue is empty.
PNG_MAIN_THREAD_DELAY = 0.001


### Reporting.

# Number of examples to visualize in the HTML visualization.
VISUALIZE_COUNT = 200
assert VISUALIZE_COUNT % BATCH_SIZE == 0

# How often to evaluate and store the training loss (minibatch count).
TRAINING_SUMMARY_PERIOD = 500

# How often to evaluate the validation loss.
VALIDATION_SUMMARY_PERIOD = 1000

# Number of examples to use for the validation loss.
VALIDATION_SUMMARY_EXAMPLE_COUNT = 1000

# How often to output the HTML visualization.
VISUALIZATION_PERIOD = 500

# How much the visualization period grows at each visualization.
VISUALIZATION_PERIOD_MULTIPLIER = 1.02	


### Indices of input features in the .darc datasets.
IND_PRIMAL_COLOR = (0, 3)
IND_DX_COLOR = (3, 6)    
IND_DY_COLOR = (6, 9)    
IND_VAR_DX = (9, 10)
IND_VAR_DY = (10, 11)
IND_VAR_PRIMAL = (11, 12)
IND_ALBEDO = (12, 15)
IND_NORMAL = (15, 18)
IND_DEPTH = (18, 19)
IND_VAR_ALBEDO = (19, 20)
IND_VAR_NORMAL = (20, 21)
IND_VAR_DEPTH = (21, 22)

# Note: Not used by our networks.
IND_DIFFUSE = (22, 25)
IND_SPECULAR = (25, 28)
IND_VAR_DIFFUSE = (28, 29)
IND_VAR_SPECULAR = (29, 30)

# Note: Not included in the datasets by default!
IND_POSITION = (None, None)
IND_VISIBILITY = (None, None)
IND_VAR_POSITION = (None, None)
IND_VAR_VISIBILITY = (None, None)


### Minibatch streaming.

def _calculate_input_count():
	'''Calculates the number of enabled inputs.'''
	
	if MODEL_INDEX < 0:
		return 0
	
	_dimension_count = 3 # Primal color always enabled.
	if model.input_enabled('gradients'):
		_dimension_count += 6

	if model.input_enabled('variances'):
		_dimension_count += 1
		if model.input_enabled('gradients'):
			_dimension_count += 2

	if model.input_enabled('albedo'):
		_dimension_count += 3
	if model.input_enabled('normal'):
		_dimension_count += 3
	if model.input_enabled('depth'):
		_dimension_count += 1
	if model.input_enabled('diffuse'):
		_dimension_count += 3
	if model.input_enabled('specular'):
		_dimension_count += 3

	if model.input_enabled('variances'):
		if model.input_enabled('albedo'):
			_dimension_count += 1
		if model.input_enabled('normal'):
			_dimension_count += 1
		if model.input_enabled('depth'):
			_dimension_count += 1
		if model.input_enabled('diffuse'):
			_dimension_count += 1
		if model.input_enabled('specular'):
			_dimension_count += 1

	return _dimension_count

def _create_index_mapping():
	'''Creates a mapping of feature index ranges from the dataset indices to the indices in the minibatch.
	   The index range of feature F in the produced minibatch is index_mapping[IND_F].'''
	   
	if MODEL_INDEX < 0:
		return {}
	   
	indices = {}
	dim = 0
	
	# Note: The order has to match that in create_darc.py.
	
	indices[IND_PRIMAL_COLOR] = (dim, dim+3)
	dim += 3
	
	if model.input_enabled('gradients'):
		indices[IND_DX_COLOR] = (dim, dim+3)
		dim += 3
		
		indices[IND_DY_COLOR] = (dim, dim+3)
		dim += 3
	
	if model.input_enabled('variances'):
		indices[IND_VAR_PRIMAL] = (dim, dim+1)
		dim += 1
		
		if model.input_enabled('gradients'):
			indices[IND_VAR_DX] = (dim, dim+1)
			indices[IND_VAR_DY] = (dim+1, dim+2)
			dim += 2
				
	if model.input_enabled('albedo'):
		indices[IND_ALBEDO] = (dim, dim+3)
		dim += 3
		
	if model.input_enabled('normal'):
		indices[IND_NORMAL] = (dim, dim+3)
		dim += 3
		
	if model.input_enabled('depth'):
		indices[IND_DEPTH] = (dim, dim+1)
		dim += 1
	
	if model.input_enabled('variances'):
		if model.input_enabled('albedo'):
			indices[IND_VAR_ALBEDO] = (dim, dim+1)
			dim += 1
		
		if model.input_enabled('normal'):
			indices[IND_VAR_NORMAL] = (dim, dim+1)
			dim += 1
			
		if model.input_enabled('depth'):
			indices[IND_VAR_DEPTH] = (dim, dim+1)
			dim += 1
			
	if model.input_enabled('diffuse'):
		indices[IND_DIFFUSE] = (dim, dim+3)
		dim += 3

	if model.input_enabled('specular'):
		indices[IND_SPECULAR] = (dim, dim+3)
		dim += 3
	
	if model.input_enabled('variances'):
		if model.input_enabled('diffuse'):
			indices[IND_VAR_DIFFUSE] = (dim, dim+1)
			dim += 1

		if model.input_enabled('specular'):
			indices[IND_VAR_SPECULAR] = (dim, dim+1)
			dim += 1
			
	return indices


_index_mapping = _create_index_mapping()
def get_minibatch_dims(feature_indices):
	'''Maps the index range (IND_FEATURE) from the dataset indices to the indices in the created minibatches.'''
	return _index_mapping[feature_indices]


# Number of network inputs in a minibatch tensor.
INPUT_COUNT = _calculate_input_count()

# Total number of features in the passed minibatch tensor: Inputs + ground truth.
TOTAL_COUNT = INPUT_COUNT + 3

# Total size of the padded image.
TOTAL_SIZE = CROP_SIZE + 2 * PAD_WIDTH

