import generator
import enqueue_renders as render


### Create the training dataset in multiple batches.

config = {
	'prefix': 'train_default', # Name of the result dataset
	'crop_size': 128,
	'crop_count': 250,
	'images_per_crop': 5,
	'spp_range': (2, 1024),
	'shutter_range': (0.0, 1.0),
	'reference_spp': 8192, # Almost enough for most scenes IF using G-PT for the ground truths.
}

# Note: The given seeds are used only for generating the sensor configurations; must match between noisy and clean.

gen = generator.Generator('../run_on_cluster/run', 'train1')
#render.render_training_crops(gen, 'bathroom',              **config, output='noisy', seed=0)
render.render_training_crops(gen, 'bathroom3',             **config, output='noisy', seed=1)
render.render_training_crops(gen, 'classroom',             **config, output='noisy', seed=2)
render.render_training_crops(gen, 'crytek_sponza',         **config, output='noisy', seed=3)
gen.close()

gen = generator.Generator('../run_on_cluster/run', 'train2')
#render.render_training_crops(gen, 'kitchen_simple',        **config, output='noisy', seed=4)
render.render_training_crops(gen, 'living-room',           **config, output='noisy', seed=5)
render.render_training_crops(gen, 'living-room-2',         **config, output='noisy', seed=6)
render.render_training_crops(gen, 'living-room-3',         **config, output='noisy', seed=7)
gen.close()

gen = generator.Generator('../run_on_cluster/run', 'train3')
render.render_training_crops(gen, 'staircase',             **config, output='noisy', seed=8)
render.render_training_crops(gen, 'staircase2',            **config, output='noisy', seed=9)
render.render_training_crops(gen, 'bathroom2',             **config, output='noisy', seed=10)  # Only for validation
#render.render_training_crops(gen, 'bookshelf_rough2',      **config, output='noisy', seed=11)  # Only for validation
render.render_training_crops(gen, 'new_kitchen_animation', **config, output='noisy', seed=12)  # Only for validation
gen.close()

gen = generator.Generator('../run_on_cluster/run', 'train4')
#render.render_training_crops(gen, 'bathroom',              **config, output='clean', seed=0)
render.render_training_crops(gen, 'bathroom3',             **config, output='clean', seed=1)
render.render_training_crops(gen, 'classroom',             **config, output='clean', seed=2)
render.render_training_crops(gen, 'crytek_sponza',         **config, output='clean', seed=3)
#render.render_training_crops(gen, 'kitchen_simple',        **config, output='clean', seed=4)
render.render_training_crops(gen, 'living-room',           **config, output='clean', seed=5)
render.render_training_crops(gen, 'living-room-2',         **config, output='clean', seed=6)
render.render_training_crops(gen, 'living-room-3',         **config, output='clean', seed=7)
render.render_training_crops(gen, 'staircase',             **config, output='clean', seed=8)
render.render_training_crops(gen, 'staircase2',            **config, output='clean', seed=9)
render.render_training_crops(gen, 'bathroom2',             **config, output='clean', seed=10)  # Only for validation
#render.render_training_crops(gen, 'bookshelf_rough2',      **config, output='clean', seed=11)  # Only for validation
render.render_training_crops(gen, 'new_kitchen_animation', **config, output='clean', seed=12)  # Only for validation
gen.close()


### Create the test datasets.

# Most of the scenes define a camera fly-by animation, and 'crop_count' specifies how many frames are rendered from that animation range.

# With gradients.
config = {
	'prefix': 'test_gpt', # Name of the result dataset
	'crop_count': 2,
	'images_per_crop': 12,
	'reference_spp': 8192, # Almost enough for most scenes IF using G-PT for the ground truths.
}

gen = generator.Generator('../run_on_cluster/run', 'test1')
render.render_test_images(gen, 'new_kitchen_animation', **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=50)
render.render_test_images(gen, 'new_bedroom',           **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=51)
render.render_test_images(gen, 'new_dining_room',       **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=52)
render.render_test_images(gen, 'new_kitchen_dof',       **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=53)
#render.render_test_images(gen, 'bookshelf_rough2',      **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=54)
render.render_test_images(gen, 'sponza',                **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=55)
render.render_test_images(gen, 'running_man',           **config, output='noisy', shutter_range=(1.0, 1.0), animation_length=192, seed=56, xml_sequence='batch%05d.xml')
gen.close()

gen = generator.Generator('../run_on_cluster/run', 'test2')
render.render_test_images(gen, 'new_kitchen_animation', **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=50)
render.render_test_images(gen, 'new_bedroom',           **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=51)
render.render_test_images(gen, 'new_dining_room',       **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=52)
render.render_test_images(gen, 'new_kitchen_dof',       **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=53)
#render.render_test_images(gen, 'bookshelf_rough2',      **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=54)
render.render_test_images(gen, 'sponza',                **config, output='clean', shutter_range=(0.0, 0.0), animation_length=120, seed=55)
render.render_test_images(gen, 'running_man',           **config, output='clean', shutter_range=(1.0, 1.0), animation_length=192, seed=56, xml_sequence='batch%05d.xml')
gen.close()

# Without gradients.
config = {
	'prefix': 'test_pt', # Name of the result dataset
	'crop_count': 2,
	'images_per_crop': 12,
	'reference_spp': 8192, # Almost enough for most scenes IF using G-PT for the ground truths.
}

# Note: new_kitchen_dof could probably use 2x or 4x more samples.

gen = generator.Generator('../run_on_cluster/run', 'test3')
render.render_test_images(gen, 'new_kitchen_animation', **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=50, use_pt=True)
render.render_test_images(gen, 'new_bedroom',           **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=51, use_pt=True)
render.render_test_images(gen, 'new_dining_room',       **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=52, use_pt=True)
render.render_test_images(gen, 'new_kitchen_dof',       **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=53, use_pt=True)
#render.render_test_images(gen, 'bookshelf_rough2',      **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=54, use_pt=True)
render.render_test_images(gen, 'sponza',                **config, output='noisy', shutter_range=(0.0, 0.0), animation_length=120, seed=55, use_pt=True)
render.render_test_images(gen, 'running_man',           **config, output='noisy', shutter_range=(1.0, 1.0), animation_length=192, seed=56, xml_sequence='batch%05d.xml', use_pt=True)
gen.close()
