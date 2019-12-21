#!/usr/bin/python
import sys
import os
import string
import datetime
import subprocess
from optparse import OptionParser


try:
	import psutil
except ImportError as ex:
	print("Could not import psutil. Try: python3 -m pip install psutil")
	

CPUS_PER_TASK = 8 # Core count for renders.

	
# Configure commandline parameters.
parser = OptionParser(usage="usage: %prog [config1 (config2 ...)]")
(options, args) = parser.parse_args()


def runBatch(config, batch_index):
	config_name = "{0}__{1}".format(config, batch_index)

	# Check that the config files exist.
	for path in ["configs/%s_parameters.cfg" % config_name,
				"configs/%s_store.cfg" % config_name,
				"configs/%s_tasks.cfg" % config_name]:
		if not os.path.isfile(path):
			print('Config "%s" not found!' % config_name)
			sys.exit(1)
	
	
	# Run jobs.
	tasks_file = "configs/%s_tasks.cfg" % config_name
	job_count = len(open(tasks_file).readlines())

	psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
	
	for parameters in open(tasks_file).readlines():
		command = "python task_run.py {} {} {}".format(config_name, CPUS_PER_TASK, parameters.strip())
		print(command)
		subprocess.call(command, shell=True);


# Run batches.
for config in args:
	runBatch(config, 0)
	runBatch(config, 1)
	runBatch(config, 2)
