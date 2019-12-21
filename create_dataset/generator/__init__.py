# File originally from Temporal Gradient-Domain Path Tracing, but simplified.

import os
import copy
import sys
import math


def formatKeyValue(key, value):
    if value is True:
        return "{}".format(key)
    elif value is False:
        return "no" + key[0].capitalize() + key[1:]
    else:
        return "{}{}".format(key, value)
    
def getDefaultTaskName(scene, integrator, spp, shutter_time, frame_count, render_parameters = {}):
    task_name = '{}-{}frames-shutter{}'.format(scene, frame_count, shutter_time)
    
    custom_parameter_strings = [formatKeyValue(key, value) for key, value in render_parameters.items()]
    task_name = '-'.join([task_name] + sorted(custom_parameter_strings))
    
    task_name += "-{}spp-{}".format(spp, integrator)
    
    return task_name
        
def makeParentDirectories(path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    
class Batch:
    def __init__(self, cluster_path, config_name, batch_index):
        self.index = 0 # Index of the current job in the batch.
        self.parameter_file_path = '{}/configs/{}__{}_parameters.cfg'.format(cluster_path, config_name, batch_index)
        self.task_file_path = '{}/configs/{}__{}_tasks.cfg'.format(cluster_path, config_name, batch_index)
        self.store_file_path = '{}/configs/{}__{}_store.cfg'.format(cluster_path, config_name, batch_index)
        
        # Prepare output data.
        self.parameters = []
        self.tasks = []
        self.stores = []
        
    def addJob(self, parameters, outdir):
        '''Adds a render job to be executed.'''
        self.parameters.append(parameters + '\n')
        self.tasks.append('{}\n'.format(self.index))
        self.stores.append(outdir + '\n')
        self.index += 1
    
    def empty(self):
        '''Returns whether the Batch is empty.'''
        return not self.tasks
    
    def close(self):
        # Output the files.

        # Make the required directories.
        makeParentDirectories(self.parameter_file_path)
        makeParentDirectories(self.task_file_path)
        makeParentDirectories(self.store_file_path)
        
        # Output the files.
        parameter_file = open(self.parameter_file_path, "w", newline='\n')
        task_file = open(self.task_file_path, "w", newline='\n')
        store_file = open(self.store_file_path, "w", newline='\n')
        
        for parameter, task, store in zip(self.parameters, self.tasks, self.stores):
            parameter_file.write(parameter)
            task_file.write(task)
            store_file.write(store)
                    
        store_file.close()
        task_file.close()
        parameter_file.close()


class Generator:
    BATCH_DEPENDENCY_TARGET = 0 # Batch to which something in BATCH_DEPENDENT depends on.
    BATCH_INDEPENDENT = 1       # Batch whose jobs are independent.
    BATCH_DEPENDENT = 2         # Batch whose jobs depend on BATCH_DEPENDENCY_TARGET.
    
    def __init__(self, cluster_path, config_name):
        # Read sys.argv by default. This is dirty but keeps the user clean in the usual use case.
            
        self.config_name = config_name
        self.cluster_path = cluster_path   # Path to the directory that's copied to the cluster.
        
        # Prepare the batch queues.
        self.batches = [
            Batch(self.cluster_path, self.config_name, 0),
            Batch(self.cluster_path, self.config_name, 1),
            Batch(self.cluster_path, self.config_name, 2)
        ]
                
    # Controls scheduling of render jobs.
    def queueRender(self, task, scene, outdir, seed, integrator, spp, frame_start_time, shutter_time, batch=BATCH_INDEPENDENT, xml_sequence='', blocksize=32, loop_length=0.0, custom={}):
        '''Queues a single render with the given parameters.'''
        # Validate parameters.
        if type(spp) != int:
            print("Error: Task \"{}\": Non-integral sample count {}".format(task, spp))
            sys.exit(1)
            
        # Convert custom parameters.
        converted_custom = {}
        for key, value in custom.items():
            if value is True:
                value = 'true'
            elif value is False:
                value = 'false'
            converted_custom[key] = value
        
        # Normalize time for sequenced scenes.
        if loop_length:
            frame_start_time = frame_start_time % loop_length
        
        if xml_sequence:
            frame_number = math.floor(frame_start_time)
            frame_start_time -= frame_number
            
            scene_input_xml = 'scenes/{}/{}'.format(scene, xml_sequence % frame_number)
        else:
            scene_input_xml = 'scenes/{}/batch.xml'.format(scene)
        
        # Construct the command line.
        commandline = '{} -b {} -Dsampler=deterministic -Dintegrator={} -Dseed={} -Dspp={} -DshutterOpen={} -DshutterClose={} {}'.format(
            scene_input_xml,
            blocksize,
            integrator,
            seed,
            spp,
            frame_start_time,
            frame_start_time + shutter_time,
            " ".join(["-D{}={}".format(key, value) for key, value in converted_custom.items()])
        )
        
        self.batches[batch].addJob(commandline, outdir)     
    
    def queueGPT(self, scene, seed, spp, frame_time, shutter_time, frame_count, frame = 0,
                 render_parameters = {},
                 seed_increment = 1, xml_sequence='', loop_length = 0.0,
                 task_name = ''):
        '''Convenience function for rendering an animation with G-PT.'''
        if not task_name:
            task_name = getDefaultTaskName(scene, 'gpt', spp, shutter_time, frame_count, render_parameters)

        FRAME_TEMPLATE = task_name + '/frame%03d_seed%d'
                
        # Queue frames to be rendered.
        for i in range(frame_count):
            current_time = (frame + i) * frame_time
            current_seed = seed + i * seed_increment
            
            # Render with seed i and store the results.
            self.queueRender(
                scene = scene,
                outdir = FRAME_TEMPLATE % (i, i),
                seed = current_seed,
                integrator = 'gpt',
                spp = spp,
                frame_start_time = current_time,
                shutter_time = shutter_time,
                task = task_name,
                xml_sequence = xml_sequence,
                custom = render_parameters,
                loop_length = loop_length
            )
        
    def close(self):
        '''Closes all files and shuts down the generator.'''
        for batch in self.batches:
            batch.close()
        
        print('Generated config "{}".'.format(self.config_name))
        