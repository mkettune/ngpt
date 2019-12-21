import pdb
import os
import io
import concurrent.futures
import argparse
import zipfile

import numpy as np

import pfm
import OpenEXR

import sys
sys.path.insert(0, '../reconstruct')
import darc


parser = argparse.ArgumentParser()
parser.add_argument('--in-directory', type=str, required=True, help='the directory which contains the rendered scenes')
parser.add_argument('--out-directory', type=str, required=True, help='the directory where the Darcs are written')
parser.add_argument('--scene-name', type=str, required=True, help='name of the scene (in the rendering batch system)')
parser.add_argument('--task-prefix', type=str, required=True, help='prefix of the results in the cluster results directory')
parser.add_argument('--gt-prefix', type=str, default='', help='prefix of the results used for ground-truths in the cluster results directory')
parser.add_argument('--crop-count', type=int, required=True, help='number of rendered crops for the scene')
parser.add_argument('--images-per-crop', type=int, required=True, help='number of rendered images for each crop, in addition to the ground truth image')
parser.add_argument('--ground-truth', type=str, default='gpt', help='whether to use the screened Poisson L1 results (\'gpt\') or the more noisy primal images (\'pt\') as the target images')
args = parser.parse_args()

assert args.ground_truth in ('gpt', 'pt')


OUT_DIR = args.out_directory
IN_DIR = args.in_directory


def readExrFromZip(zip_file, image_name):
	import Imath
	
	exr_in = io.BytesIO(zip_file.read(image_name))
	exr_file = OpenEXR.InputFile(exr_in)
	
	exr_header = exr_file.header()
	H, W = exr_header['dataWindow'].max.y + 1, exr_header['dataWindow'].max.x + 1
	assert all((exr_header['channels'][channel].type == Imath.PixelType(OpenEXR.HALF) for channel in ('R', 'G', 'B')))
	
	R, G, B = exr_file.channels(['R', 'G', 'B'])
	image_R = np.frombuffer(R, dtype=np.float16).reshape([H, W])
	image_G = np.frombuffer(G, dtype=np.float16).reshape([H, W])
	image_B = np.frombuffer(B, dtype=np.float16).reshape([H, W])
	image = np.stack([image_R, image_G, image_B], axis=2)
	exr_file.close()
	
	return image.astype(np.float32)

def create_darc(scene_name, task_prefix, gt_prefix, ground_truth, crop_count, images_per_crop):
	darc_path = OUT_DIR + "/{}-{}-{}-{}.darc".format(task_prefix, scene_name, crop_count, images_per_crop)
	
	def clampToZero(primal, dx=None, dy=None):
		return np.maximum(primal, 0.0), dx, dy

	def combineNoisyData(crop_index, image_index):
		in_dir = os.path.join(IN_DIR, "{}-{}-{}-{}".format(task_prefix, scene_name, crop_index, image_index), 'frame000_seed0')
		in_zip = zipfile.ZipFile(os.path.join(in_dir, 'data.zip'))
		
		def loadNoisyData(image_type):
			return readExrFromZip(in_zip, 'image-{}.exr'.format(image_type))

		primal = loadNoisyData('primal')
		dx = loadNoisyData('dx')
		dy = loadNoisyData('dy')
				
		primal, dx, dy = clampToZero(primal, dx, dy)
				
		albedo = loadNoisyData('albedo')
		normal = loadNoisyData('normal')
		depth = loadNoisyData('depth')
		depth = depth[:, :, 0:1]

		diffuse = loadNoisyData('diffuse')
		specular = loadNoisyData('specular')
				
		var_primal = loadNoisyData('var-primal')
		var_primal = np.mean(var_primal, axis=2, keepdims=True)
		
		var_dx = loadNoisyData('var-dx')
		var_dx = np.mean(var_dx, axis=2, keepdims=True)
		
		var_dy = loadNoisyData('var-dy')
		var_dy = np.mean(var_dy, axis=2, keepdims=True)
		
		var_albedo = loadNoisyData('var-albedo')
		var_albedo = np.mean(var_albedo, axis=2, keepdims=True)
		
		var_normal = loadNoisyData('var-normal')
		var_normal = np.mean(var_normal, axis=2, keepdims=True)
		
		var_depth = loadNoisyData('var-depth')
		var_depth = var_depth[:, :, 0:1]
		
		var_diffuse = loadNoisyData('var-diffuse')
		var_diffuse = np.mean(var_diffuse, axis=2, keepdims=True)
		
		var_specular = loadNoisyData('var-specular')
		var_specular = np.mean(var_specular, axis=2, keepdims=True)
		
		image = np.concatenate([
			primal, dx, dy,
			var_dx, var_dy, var_primal,
			albedo, normal, depth,
			var_albedo, var_normal, var_depth,
			diffuse, specular,
			var_diffuse, var_specular
		], axis=2)
		
		return image
		
	def combineCleanData(crop_index):
		in_dir = os.path.join(IN_DIR, "{}-{}-{}-{}".format(gt_prefix, scene_name, crop_index, 'clean'), 'frame000_seed0')
		print(in_dir)
		in_zip = zipfile.ZipFile(os.path.join(in_dir, 'data.zip'))
		
		def loadCleanData(image_type):
			if image_type == 'final':
				with open(os.path.join(in_dir, 'image-final.pfm'), 'rb') as f:
					image, _ = pfm.load_pfm(f)
					return image[::-1,:,:].astype(dtype=np.float32)

			return readExrFromZip(in_zip, 'image-{}.exr'.format(image_type))
		
		dx = loadCleanData('dx')
		dy = loadCleanData('dy')
		
		if ground_truth == 'gpt':
			target = loadCleanData('final')
		elif ground_truth == 'pt':
			target = loadCleanData('primal')
		else:
			raise Exception('Unknown target image type.')
			
		target, dx, dy = clampToZero(target, dx, dy)

		var_primal = loadCleanData('var-primal')
		var_primal = np.mean(var_primal, axis=2, keepdims=True)
		
		var_dx = loadCleanData('var-dx')
		var_dx = np.mean(var_dx, axis=2, keepdims=True)
		
		var_dy = loadCleanData('var-dy')
		var_dy = np.mean(var_dy, axis=2, keepdims=True)
		
		albedo = loadCleanData('albedo')

		normal = loadCleanData('normal')
		
		depth = loadCleanData('depth')
		depth = depth[:, :, 0:1]

		diffuse = loadCleanData('diffuse')
		specular = loadCleanData('specular')
		
		var_albedo = loadCleanData('var-albedo')
		var_albedo = np.mean(var_albedo, axis=2, keepdims=True)
		
		var_normal = loadCleanData('var-normal')
		var_normal = np.mean(var_normal, axis=2, keepdims=True)
		
		var_depth = loadCleanData('var-depth')
		var_depth = var_depth[:, :, 0:1]
		
		var_diffuse = loadCleanData('var-diffuse')
		var_diffuse = np.mean(var_diffuse, axis=2, keepdims=True)
		
		var_specular = loadCleanData('var-specular')
		var_specular = np.mean(var_specular, axis=2, keepdims=True)
		
		image = np.concatenate([
			target, dx, dy,
			var_dx, var_dy, var_primal,
			albedo, normal, depth,
			var_albedo, var_normal, var_depth,
			diffuse, specular,
			var_diffuse, var_specular
		], axis=2)
		
		return image
		
	def prepare():
		# Create the archives.
		os.makedirs(OUT_DIR, exist_ok=True)
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
			with darc.DataArchive(darc_path, 'w') as archive:
			
				archive_elements = []
				
				def handle_crop_index(crop_index):
					images = []

					for image_index in range(images_per_crop):
						images.append(combineNoisyData(crop_index, image_index))
                    
					images.append(combineCleanData(crop_index))
						
					image_stack = np.stack(images)
					return image_stack.astype(np.float16)
					
				for crop_index in range(crop_count):
					archive_elements.append(executor.submit(handle_crop_index, crop_index))
				
				for i, archive_element in enumerate(archive_elements):
					archive.append(archive_element.result(), chunks=(1, 128, 128, 3))
					print("Adding crop {}/{} to archive.".format(i+1, crop_count))
					
	prepare()

def process(scene_name, task_prefix, gt_prefix, ground_truth, crop_count, images_per_crop):
	print("Processing scene '{}'.".format(scene_name))
	create_darc(scene_name, task_prefix, gt_prefix, ground_truth, crop_count, images_per_crop)
	print("Finished scene '{}'.".format(scene_name))


if __name__ == '__main__':	
	gt_prefix = args.gt_prefix if args.gt_prefix else args.task_prefix
	
	process(args.scene_name, args.task_prefix, gt_prefix, args.ground_truth, args.crop_count, args.images_per_crop)
