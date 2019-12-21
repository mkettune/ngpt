import os
import io
import sys
import subprocess
import shutil
import concurrent.futures

import numpy as np
import OpenEXR
import pfm

import zipfile


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
		
def savePfm(path, image):
	with open(path, 'wb') as file:
		pfm.save_pfm(file, image[::-1,:,:])

		
def reconstruct(input_root, temp_root, poisson_exe, i, in_dir, in_dirs):
	temp_dir = os.path.join(temp_root, "{:06d}".format(i))
	current_in_dir = os.path.join(input_root, in_dir, 'frame000_seed0')

	# Skip if already reconstructed.
	reconst_file = os.path.join(current_in_dir, 'image-final.pfm')
	if os.path.exists(reconst_file):
		print("{}/{}: Already reconstructed.".format(i, len(in_dirs)))
		return
	
	zip_in_path = os.path.join(current_in_dir, 'data.zip')

	# Copy input data to temp directory for L1 reconstruction.
	zip_in = zipfile.ZipFile(zip_in_path)
	
	os.makedirs(temp_dir, exist_ok=True)

	def extractPfm(image_name):
		try:
			image = readExrFromZip(zip_in, image_name+'.exr')
		except KeyError as ex:
			raise Exception("Failed render detected: '{}' lacks '{}'.".format(zip_in_path, image_name+".exr"))

		savePfm(os.path.join(temp_dir, image_name+'.pfm'), image)
		
	print("{}/{}: Extracting '{}' to '{}'.".format(i, len(in_dirs), zip_in_path, temp_dir))

	extractPfm('image-primal')
	extractPfm('image-dx')
	extractPfm('image-dy')
	extractPfm('image-direct')
	
	# Reconstruct.
	print("{}/{}: Running screened Poisson reconstruction in '{}'.".format(i, len(in_dirs), temp_dir))
	process = subprocess.Popen([poisson_exe, '-dx', '{}/image-dx.pfm'.format(temp_dir), '-alpha', '0.2', '-config', 'L1D', '-nopngout'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	process.communicate()

	shutil.copy2(os.path.join(temp_dir, 'image-final.pfm'), os.path.join(current_in_dir, 'image-final.pfm'))
	
	shutil.rmtree(temp_dir)
	
def reconstruct_all(input_root, temp_root, poisson_exe):
	if not os.path.exists(poisson_exe):
		raise Exception("Poisson executable '{}' not found.".format(poisson_exe))
	
	# Screened Poisson reconstruct the reference images.
	in_dirs = os.listdir(input_root)

	print("Reconstructing clean images in \"{}\" for additional noise removal in ground truth images.".format(input_root))
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		tasks = []
		for i, in_dir in enumerate(in_dirs):
			if not in_dir.endswith('-clean'):
				continue

			tasks.append(executor.submit(reconstruct, input_root, temp_root, poisson_exe, i+1, in_dir, in_dirs))
		
		for task in concurrent.futures.as_completed(tasks):
			task.result()
