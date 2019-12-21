#!/usr/bin/python
import sys
import os
import string
import datetime
import subprocess
import parse
from optparse import OptionParser


def toTime(hours = 0, minutes = 0, seconds = 0):
	return int(hours * 3600 + minutes * 60 + seconds)

def formatRenderTime(time_seconds):
	return "%02d:%02d:%02d" % (int(time_seconds / 3600), int((time_seconds / 60)) % 60, int(time_seconds % 60))

	
	
DEFAULT_PARTITION = 'short'
DEFAULT_RENDER_TIME = toTime(minutes = 60)

NOTE = 'mts'
OUT_DIRECTORY = "%s/out" % os.environ['WRKDIR']

MEMORY = 2 * 1024

	
# Configure commandline parameters.
parser = OptionParser(usage="usage: %prog [config] arg1 arg2")

parser.add_option('-p', '--partition', dest='partition',
                  type='string', default=DEFAULT_PARTITION,
                  metavar='PARTITION',
				  help='Add task to partition PARTITION. Possibilities: play, short, batch.')
parser.add_option('-t', '--hours', dest='hours',
                  type='float', default=0.0,
                  metavar='HOURS',
				  help='Add HOURS hours to execution time of a single render.')
parser.add_option('-m', '--minutes', dest='minutes',
                  type='float', default=0.0,
                  metavar='MINUTES',
                  help='Add MINUTES minutes to execution time of a single render.')

parser.add_option('-s', '--single', dest='single', action='store_true', default=False,
                  help='Run a single configuration batch without an index suffix.')
parser.add_option('-d', '--depend', dest='dependency', default=-1,
                  help='Run the batch only after the given batch has finished successfully.')

parser.add_option('-c', '--cpus', dest='cpus',
                  type='int', default=12,
                  help='How many CPUs to run with.')


(options, args) = parser.parse_args()


# Set defaults.
render_time = 0
config = 'default'

# Apply commandline arguments.
if not options.hours and not options.minutes:
	render_time = DEFAULT_RENDER_TIME
else:
	if options.hours:
		render_time += toTime(hours = float(options.hours))
	if options.minutes:
		render_time += toTime(minutes = float(options.minutes))

partition = options.partition

run_single = options.single
dependency = options.dependency

cpus_per_task = options.cpus


if args:
	config = args[0]


	
# Create log directory.

# Read the next log index.
try:
	log_index_file = open(OUT_DIRECTORY + '/log_index.cfg')
	log_index = 1 + int(log_index_file.readlines()[0])
	log_index_file.close()
except:
	log_index = 1
	
# Update the next log index.
log_index_file = open(OUT_DIRECTORY + '/log_index.cfg', 'wt')
log_index_file.write('{0}'.format(log_index))
log_index_file.close()
	
# Create the directory.
LOG_DIRECTORY = '%s/log/%05d' % (OUT_DIRECTORY, log_index)

if not os.path.isdir(LOG_DIRECTORY):
	os.makedirs(LOG_DIRECTORY)
if not os.path.isdir(OUT_DIRECTORY):
	os.makedirs(OUT_DIRECTORY)

	
def configName(config, batch_index, run_single = False):
	'''Returns the name of a config basing on its basename and batch index.'''
	if not run_single:
		return "{0}__{1}".format(config, batch_index)
	else:
		return "{0}".format(config)
	
def batchEmpty(config, batch_index, run_single = False):
	'''Returns whether a batch is empty.'''
	path = "configs/%s_tasks.cfg" % configName(config, batch_index, run_single)
	if not os.path.isfile(path):
		return True
	
	task_count = len(open(path, 'r').readlines())
	return task_count == 0
	
	
def runBatch(config, batch_index, dependency = -1, run_single = False):
	config_name = configName(config, batch_index, run_single)
	short_name = "{0}-{1}".format(NOTE, config_name)
	long_name = "{0}-{1}_{2}".format(NOTE, config_name, datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
	
	# Check that the files exist.
	for path in ["configs/%s_parameters.cfg" % config_name,
				"configs/%s_store.cfg" % config_name,
				"configs/%s_tasks.cfg" % config_name]:
		if not os.path.isfile(path):
			print('Config "%s" not found!' % config_name)
			sys.exit(1)
		
	# Generate the template to show to the user.
	long_name = "%s_%s_%s" % (NOTE, config, datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))

	# Show user info.
	user_info = string.Template('''
==========================================
    output directory: $outDir
             jobname: ${shortNameOfScript}
     max render time: ${renderTime}
                note: ${note}
       log directory: ${logDir}
    output directory: ${outDir}
           partition: ${partition}
------------------------------------------
''').substitute({
	'outDir' : OUT_DIRECTORY,
	'shortNameOfScript' : short_name,
	'renderTime' : formatRenderTime(render_time),
	'note' : NOTE,
	'logDir' : LOG_DIRECTORY,
	'outDir' : OUT_DIRECTORY,
	'partition' : partition,
	'config' : config
})

	print(user_info)


	# Generate and launch the scripts.
	tasks_file = "configs/%s_tasks.cfg" % config_name
	job_count = len(open(tasks_file).readlines())

	########################################################
	# Note, the following codes are interpreted in the script variable
	#       \\     backslash
	#       \a     alert (BEL)
	#       \b     backspace
	#       \c     produce no further output
	#       \e     escape
	#       \f     form feed
	#       \n     new line
	#       \r     carriage return
	#       \t     horizontal tab
	#       \v     vertical tab
	#       \0NNN  byte with octal value NNN (1 to 3 digits)
	#       \xHH   byte with hexadecimal value HH (1 to 2 digits)

	script = string.Template(
'''#!/bin/bash
#SBATCH -t $renderTime
#SBATCH -p $partition
#SBATCH --cpus-per-task=$CPUsPerTask
#SBATCH -J ${shortNameOfScript}
#SBATCH --array=1-$jobCount
#SBATCH --mem $memory

#SBATCH -e $logDir/err_%A_%a
#SBATCH -o $logDir/out_%A_%a

echo "$info"

echo "number of cores: $$(cat /proc/cpuinfo | grep -c 'model name')"
echo "$$(cat /proc/cpuinfo | grep -m 1 'model name')"

echo "hostname: "; hostname
echo "outDir: $outDir"

echo -e "\\n\\n"

parameters=$$(head -n $${SLURM_ARRAY_TASK_ID} configs/${config}_tasks.cfg | tail -1);
command="python task_run.py $config $CPUsPerTask $$parameters";

echo " "
echo "command=\"$$command\"";
echo " "
export MTS_IDENTIFIER=%A_%a
time eval $$command
''').substitute({
	'outDir' : OUT_DIRECTORY,
	'renderTime' : formatRenderTime(render_time),
	'shortNameOfScript' : short_name,
	'note' : NOTE,
	'logDir' : LOG_DIRECTORY,
	'outDir' : OUT_DIRECTORY,
	'partition' : partition,
	'info' : user_info,
	'CPUsPerTask' : cpus_per_task,
	'jobCount' : job_count,
	'config' : config_name,
	'memory': MEMORY,
})

	# Output the batch script for debugging.
	#print(script)

	# Run sbatch, pipe the script to stdin and output stdout as it becomes available.
	
	command = ["sbatch"]
	if dependency >= 0:
		command.append("--depend=afterok:{0}".format(dependency))
	
	process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	process.stdin.write(script.encode('utf-8'))
	process.stdin.close()

	print("\nSlurm response:\n")

	job_id = -1
	
	while True:
		line = process.stdout.readline()
		print(line.decode('utf-8'))
		
		try:
			results = parse.parse("Submitted batch job {:d}", line)
			job_id = results[0]
		except Exception as ex:
			pass
	
		if not line:
			break
	
	return job_id


# Run the batches.

if run_single:
	runBatch(config, 0, dependency=dependency, run_single=True)
else:
	job_id = -1

	if not batchEmpty(config, 0):
		job_id = runBatch(config, 0, dependency=dependency)

	if not batchEmpty(config, 1):
		runBatch(config, 1, dependency=dependency)

	if job_id > 0:
		if not batchEmpty(config, 2):
			runBatch(config, 2, dependency=job_id)

