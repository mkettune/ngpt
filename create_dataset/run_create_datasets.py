import subprocess
import os

import sys
sys.path.insert(0, '../reconstruct')
import config


IN_DIRECTORY = r'../run_on_cluster/run/results'
OUT_DIRECTORY = config.DATASET_DIR


def create_dataset(scene_name, task_prefix, crop_count, images_per_crop, gt_prefix=None):
	if gt_prefix is None:
		gt_prefix = task_prefix
	
	# Combine the rendered images into .darc datasets.
	COMMAND = [
		'python', 'create_darc.py',
		'--in-directory', IN_DIRECTORY,
		'--out-directory', OUT_DIRECTORY,
		'--task-prefix', task_prefix,
		'--gt-prefix', gt_prefix,
		'--crop-count', str(crop_count),
		'--images-per-crop', str(images_per_crop),
		'--ground-truth', 'gpt', # Use G-PT for ground truths to remove the residual noise. Alternative: 'pt'.
		'--scene-name', scene_name,
	]

	print("Creating dataset for '{}' to directory '{}'.".format(scene_name, OUT_DIRECTORY))
	process = subprocess.Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr)
	process.communicate()


# Create the training datasets.
TASK_PREFIX = 'train_default'
CROP_COUNT = 250 # Crops per scene
IMAGES_PER_CROP = 5 # SPPs per crop

#create_dataset('bathroom', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('bathroom3', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('classroom', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('crytek_sponza', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
#create_dataset('kitchen_simple', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('living-room', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('living-room-2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('living-room-3', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('staircase', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('staircase2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)

# Not used for training.
create_dataset('bathroom2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
#create_dataset('bookshelf_rough2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('new_kitchen_animation', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)


# Create the test datasets for gradient reconstruction.
TASK_PREFIX = 'test_gpt'
CROP_COUNT = 2 # Crops per scene
IMAGES_PER_CROP = 12 # SPPs per crop

create_dataset('new_dining_room', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('new_kitchen_animation', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('new_kitchen_dof', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
#create_dataset('bookshelf_rough2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('new_bedroom', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('new_dining_room', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('sponza', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)
create_dataset('running_man', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP)

# Create the test datasets for primal reconstruction with 2.5x samples.
TASK_PREFIX = 'test_pt'
GT_PREFIX = 'test_gpt' # Reuse the same ground-truth images.
CROP_COUNT = 2 # Crops per scene
IMAGES_PER_CROP = 12 # SPPs per crop

create_dataset('new_dining_room', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('new_kitchen_animation', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('new_kitchen_dof', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
#create_dataset('bookshelf_rough2', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('new_bedroom', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('new_dining_room', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('sponza', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
create_dataset('running_man', TASK_PREFIX, CROP_COUNT, IMAGES_PER_CROP, gt_prefix=GT_PREFIX)
