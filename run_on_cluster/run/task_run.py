import sys
import os
import shutil
import subprocess
import time
import zipfile
import concurrent.futures
import numpy as np


MITSUBA = "dist/mitsuba"


if len(sys.argv) < 4:
	print("Usage: python {0} [config name] [cpu count] [batch index]".format(sys.argv[0]))
	sys.exit(1)

# Read configuration.
CONFIG_NAME = sys.argv[1]
CPU_COUNT = int(sys.argv[2])
BATCH_INDEX = int(sys.argv[3])

PARAMETER_FILE = "configs/{0}_parameters.cfg".format(CONFIG_NAME)
STORE_FILE = "configs/{0}_store.cfg".format(CONFIG_NAME)

PARAMETERS = open(PARAMETER_FILE).readlines()[BATCH_INDEX].strip()
STORE_DIRECTORY = os.path.join("results", open(STORE_FILE).readlines()[BATCH_INDEX]).strip()

MTS_IDENTIFIER = os.getenv('MTS_IDENTIFIER', '')


# Create the result directory.
try:
	os.makedirs(STORE_DIRECTORY)
except:
	pass

# Run the task with the given configuration.
if CPU_COUNT > 0:
	cpu_flag = "-p {0}".format(CPU_COUNT)
else:
	cpu_flag = ""
	
commandline = MITSUBA + " -z -o {0}/image {1} {2}".format(STORE_DIRECTORY, PARAMETERS, cpu_flag)


# Store the command line.
file = open(os.path.join(STORE_DIRECTORY, "info.txt"), "w")
file.write("Command line: ")
file.write(commandline)
file.write("\n")

# Store the job identifier.
file.write("Job identifier: ")
file.write(MTS_IDENTIFIER)
file.write("\n")
file.close()

# Write the config name.
file = open(os.path.join(STORE_DIRECTORY, "cluster_batch.txt"), "w")
file.write("{0}".format(CONFIG_NAME))
file.close()

start_time = time.time()
subprocess.call(commandline.split(" "))
end_time = time.time()

# Rename the results.
for file in os.listdir(STORE_DIRECTORY):
	source_file = os.path.join(STORE_DIRECTORY, file)
	
	target_file = file.replace("_iter1_0", "")	
	target_file = os.path.join(STORE_DIRECTORY, target_file)
	
	if file.startswith("image"):
		shutil.move(source_file, target_file)

# Copy the batch identifier.
shutil.copy2(os.path.join(STORE_DIRECTORY, "cluster_batch.txt"), os.path.join(STORE_DIRECTORY, os.pardir, "cluster_batch.txt"))

# Store the elapsed time.
file = open(os.path.join(STORE_DIRECTORY, "info.txt"), "a")
file.write("Rendering time: ")
file.write("{0}".format(end_time - start_time))
file.write("\n")
file.close()

# Create the timestamps.
file = open(os.path.join(STORE_DIRECTORY, "timestamp.txt"), "w");
file.write("{0}".format(int(time.time())))
file.close()
shutil.copy2(os.path.join(STORE_DIRECTORY, "timestamp.txt"), os.path.join(STORE_DIRECTORY, os.pardir, "timestamp.txt"))


# Store the files in a ZIP, as some cluster environments limit the number of individual files.
file_list = os.listdir(STORE_DIRECTORY)
zip_file = zipfile.ZipFile(os.path.join(STORE_DIRECTORY, 'data.zip.tmp'), 'w', compression=zipfile.ZIP_STORED)

def read_file(file):
	with open(os.path.join(STORE_DIRECTORY, file), 'rb') as in_file:
		return np.fromfile(in_file, dtype=np.uint8), file

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
	tasks = []
	for file in file_list:
		if file == 'data.zip' or file == 'data.zip.tmp':
			continue
		tasks.append(executor.submit(read_file, file))

	for task in tasks:
		file_contents, file_name = task.result()
		zip_file.writestr(file_name, file_contents)
		
zip_file.close()

# Delete the old files.
for file in file_list:
	if file == 'data.zip.tmp':
		continue
	if file.endswith('.zip') or file.endswith('.txt') or file.endswith('.exr') or file.endswith('.pfm'):
		os.unlink(os.path.join(STORE_DIRECTORY, file))

os.rename(os.path.join(STORE_DIRECTORY, 'data.zip.tmp'), os.path.join(STORE_DIRECTORY, 'data.zip'))
print("Created ZIP.")