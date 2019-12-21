## Deep Convolutional Reconstruction for Gradient-Domain Rendering

This document contains more detailed information about the source distribution.


## Installing the Python Modules

The reconstruction code requires Python 3.6 or newer and TensorFlow 1.12 or newer, compiled with GPU support. The following modules are also required: OpenEXR, psutil, imageio.

We recommend the Miniconda Python distribution for easy installation. The following contains step-by-step instructions for setting up the environment.

First, install Miniconda from https://docs.conda.io/en/latest/miniconda.html and open a new Miniconda terminal.

Next, create a new Conda environment:
>conda create -n ngpt_env<br>
>conda activate ngpt_env

Install Python 3.7 to the environment:
>conda install python=3.7

Install OpenEXR by downloading the correct OpenEXR .whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/ (e.g. cp37 for Python 3.7 and win_amd64 for 64-bit Windows) and install it:
>python -m pip install OpenEXR-1.3.2-cp37-cp37m-win_amd64.whl

Install TensorFlow with GPU support for Python 3.7. Use 'conda search tensorflow' to see the options and select the correct version with the following syntax, e.g.,
>conda install tensorflow=1.13.1=gpu_py37hbc1a9d5_0

Install psutil,
>python -m pip install psutil

Install imageio,
>conda install imageio

Ensure that importing the modules works. (If the Python console freezes at startup, press ctrl-c.)
> python<br>
> \>\>\> import tensorflow as tf<br>
> \>\>\> import OpenEXR<br>
> \>\>\> import psutil<br>
> \>\>\> import imageio


## Testing with the Pre-Trained Networks

The directory 'run_on_cluster/run/dist' contains a Windows build of Mitsuba with Gradient-Domain Path Tracing implementation modified to output the data required for the neural reconstruction. Run 'mtsgui.exe' in that directory, open one of the test scenes in 'run_on_cluster/run/scenes', and render it with the 'Gradient-Domain Path Tracer' plugin with the 'independent' or 'deterministic' sampler and the 'box' filter. Other samplers and pixel filters may or may not work but are not officially supported since they would benefit from retraining. See 'Provided Datasets and Scenes' below for more scenes.

Rendering the scene creates a set of EXR files next to the scene XML (e.g., '/path/to/scene/scene.xml'). NGPT can now be run, and will read these EXRs. Start by closing Mitsuba (cuDNN may sometimes fail to load otherwise) and go to directory 'reconstruct/'. Now run the following command:
> python run_single_mitsuba.py --config=100 --reconstruct=/path/to/scene/scene.xml

This should output the reconstructed image as the file '/path/to/scene/scene-100_ngpt_elpips_sqz.exr' along with the corresponding files in NPZ and PNG formats. The NPZ files store the raw data in NumPy compressed format and are easy to load for further processing in Python (see load_npz in image.py). The config number 100 means the pre-trained default configuration trained with gradients enabled and with the E-LPIPS (SqueezeNet) loss. The string '100_ngpt_elpips_sqz' is the name given to the model in 'config.py'.

The other provided pre-trained configurations are described below.


## List of Pre-Trained Networks

This distribution comes with four pre-trained networks for testing purposes. All networks were trained for 3 full days with a single NVIDIA V100 GPU with the default configurations in 'config.py'. The following lists the provided pre-trained configurations. See 'config.py' for details.

| Config |   Description   |
| :----: | :-------------: |
| 100 | **Default;** with gradients, E-LPIPS (Sqz) loss |
| 101 |     No gradients, E-LPIPS (Sqz) loss        |
| 102 | **Alternative;** with gradients, L1 loss    |
| 103 |        No gradients, L1 loss                |

Use the command line parameter '--config=N' to set the used model.

These networks were trained with a newer version of the source code than what was used for the paper. While the results should be very similar to the ones reported in the paper, small differences due to variance in network training are possible.


## Provided Datasets and Scenes

We provide pre-rendered datasets for training and testing. We also provide the scene descriptions for recreating the datasets. The training datasets consist of 250 pieces of random 128x128 crops, each rendered with five different sample counts. The compressed datasets are available from below and should be extracted to the directory 'dataset'. This readme also includes instructions for rebuilding the datasets if needed (see 'Creating Datasets' below).

The datasets are in the [Darc format](https://github.com/mkettune/darc/) which is similar to hdf5 but simpler. The datasets contain a multitude of featured from the renderer, of which only some are used by the network. The features that are used by default are the primal colors, gradients, and the albedo, normal, and depth auxiliary buffers. Other included but not used features are e.g. the variance buffers and the separation of color into 'diffuse' and 'specular' as described in <a href="https://doi.org/10.1145/3072959.3073708">Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings (Bako and Vogels et al. 2017)</a>.

We originally used two more scenes ('bathroom' and 'kitchen_simple') for training, and one more scene ('bookshelf_rough2') for testing. These scenes are Mitsuba ports of scenes from "Archinteriors vol. 1" from EverMotion, and are not included in this package for copyright reasons.

The following list contains the used scenes used and their purposes. See file 'readme.txt' of the scene archive (link below) for licensing information and credits. The overstruck scenes are not provided for copyright reasons.

| Directory | Used For | Scene |
| :---: | -------: | -----: |
| ~~bookself_rough2~~ | ~~Testing, Validation~~ | <a href="https://evermotion.org/shop/show_product/archinteriors-vol-1/37">~~Scene 4, Archinteriors vol. 1, EverMotion~~</a> |
| new_bedroom | Testing | <a href="https://benedikt-bitterli.me/resources/">'Bedroom' by SlykDrako,</a> |
| new_dining_room | Testing | <a href="https://benedikt-bitterli.me/resources/">'The Breakfast Room' by Wig42</a> |
| new_kitchen_animation | Testing, Validation | <a href="https://benedikt-bitterli.me/resources/">'Country Kitchen' by Jay-Artist</a> |
| new_kitchen_dof | Testing | <a href="https://benedikt-bitterli.me/resources/">'Country Kitchen' by Jay-Artist</a> |
| running_man | Testing | 'Running man' by Sampo Rask |
| sponza | Testing | 'Sponza' by Marko Dabrovic |
| ~~bathroom~~ | ~~Training~~ | <a href="https://evermotion.org/shop/show_product/archinteriors-vol-1/37">~~Scene 1, Archinteriors vol. 1, EverMotion~~</a> |
| bathroom3 | Training | <a href="https://benedikt-bitterli.me/resources/">'Salle de bain' by nacimus</a> |
| classroom | Training | <a href="https://benedikt-bitterli.me/resources/">'Japanese Classroom' by NovaZeeke</a> |
| crytek_sponza | Training | 'Crytek Sponza' by Frank Meinl, Crytek |
| ~~kitchen_simple~~ | ~~Training~~ | <a href="https://evermotion.org/shop/show_product/archinteriors-vol-1/37">~~Scene 5, Archinteriors vol. 1, EverMotion~~</a> |
| living-room | Training | <a href="https://benedikt-bitterli.me/resources/">'The Grey & White Room' by Wig42</a> |
| living-room-2 | Training | <a href="https://benedikt-bitterli.me/resources/">'The White Room' by Jay-Artist</a> |
| living-room-3 | Training | <a href="https://benedikt-bitterli.me/resources/">'The Modern Living Room' by Wig42</a> |
| staircase | Training | <a href="https://benedikt-bitterli.me/resources/">'The Wooden Staircase' by Wig42</a> |
| staircase2 | Training | <a href="https://benedikt-bitterli.me/resources/">'Modern Hall' by NewSee2l035</a> |
| bathroom2 | Validation | <a href="https://benedikt-bitterli.me/resources/">'Contemporary Bathroom' by Mareck</a> |

The terms 'testing', 'training', and 'validation' are used here in the following senses:
- Testing: Full-size images for numerical and visual analysis of the network results.
- Training: Pairs of input and output crops used for optimizing the network weights.
- Validation: Pairs of input and output crops **not** used for training, but used for visualizing the training progress.

The training and testing sets do not overlap and the scenes used for training should not be used for evaluating the network's performance.


For experimenting with the model and retraining, download the pre-rendered datasets from below and extract them into 'datasets/'. See the section 'Retraining the Model' below for further instructions.

For testing the networks and recreating the datasets, download the pre-configured Mitsuba scenes from below and extract them into 'run_on_cluster/run/scenes'.

**Download:** [Pre-rendered Datasets](https://doi.org/10.5281/zenodo.3588466) (13.1 GB)

**Download:** [Pre-configured Scenes](https://doi.org/10.5281/zenodo.3588482) (2.5 GB)


## Testing and Evaluating the Models

After downloading the datasets, you can reconstruct and evaluate the test scenes with the script 'run_results.py' in directory 'reconstruct'. This script controls reconstructing and evaluating the images in the test datasets.

You can change the used test datasets by updating the 'DATASETS' variable in 'run_results.py'.

The different options for 'run_results.py' can be displayed by running the following command:
>python run_results.py --help

The outputs will, by default, go under directory 'results/base_model/MODEL_NAME'. The output files with "crop index" N contain reconstructions from 2^N samples, or 2.5 times that (rounded up) if gradients are not enabled in the model. This is done to enable approximately equal-time comparisons.

To run the reconstruction for the images in the test dataset, run 'run_results.py' with the model's CONFIG number, e.g.,
>python run_results.py --reconstruct --config=100

This will create the neural network reconstructions into the result directories ('results/base_model/MODEL_NAME') in both NPZ and PNG formats. See file 'reconstruct/image.py' for code to open the NPZ files as NumPy arrays.

To estimate the numerical quality of the reconstructions, run 'run_results.py' with flag '--run-metrics'. The number of samples for the stochastic E-LPIPS image similarity metric (VGG version) defaults to 200 but can be increased with '--elpips-sample-count=N' for less noisy E-LPIPS estimations. This command may take some time if accurate results are requested.
>python run_results.py --config=100 --run-metrics

This command ('--run-metrics') will output a number of .csv files containing the L1, L2, RelMSE, and E-LPIPS numbers, and the standard deviation of the stochastic E-LPIPS estimate, to the directory 'results/base_model/MODEL_NAME/TEST_DATASET'. The E-LPIPS numbers typically provide the best prediction for human opinion.

Run the script 'run_results.py' with flag '--extract-dataset' to extract the input and ground truth datasets of the current model (selected with '--config') as NPZ and PNG images.
>python run_results.py --config=100 --extract-dataset

Note that the provided datasets were created with the cleaned up code and are rendered with new random seeds. Hence reconstructions from the new test datasets may differ slightly from the ones in the supplemental material.


## Retraining the Model

Modifying the model or changing the dataset requires retraining which can be a slow process: the networks should be trained for many days even with a very fast GPU. To train a model, first download the pre-made datasets by the above instructions or re-render your own datasets with the instructions below.

To train the network, first go to directory 'reconstruct'. See the available model configurations in file 'config.py' in function 'get_model_config'. Check the number of the configuration you want to train, e.g., 100 for the default NGPT configuration with E-LPIPS (SqueezeNet), or 102 for the version trained with the L1 loss. Also validate the other settings. Start the training by running 'run_training.py' with the configuration number, e.g. 'python run_training.py --config=CONFIG'. The training should be run for around 600 000 minibatches of size 20.
>cd reconstruct<br>
>python run_training.py --config=100

Note that this will overwrite the pre-trained weights for the given configuration. Duplicate the configurations to other numbers to retain the original networks.

Select the used datasets by setting the 'TRAIN_DATASET_PATHS' and 'VALIDATION_DATASET_PATHS' variables in 'config.py'. The training datasets are used for training while the validation datasets are used for displaying training progress via TensorBoard and occasional HTML reports. The TensorBoard summaries are written into directory 'tensorflow/summaries'. The training automatically creates HTML reports in 'html/result_html_CONFIG', e.g., 'html/result_html_100' where 100 is the model index as defined in 'config.py'.

A stopped training run may be continued by adding the commandline parameter '--restore'.

The training is, by default, configured to run for a given large number of minibatches which does not correspond to any given specific time. The learning rate is configured to decrease exponentially as time goes on. Be sure to edit file 'config.py' and see the instructions there before training.

The following section outlines instructions for rendering new test and training datasets for experimenting with other scenes or rendering algorithms.


## Creating Datasets

The dataset creation requires rendering a high number of input images and the corresponding high quality target / ground truth images. The images are rendered with a modified version of Mitsuba (provided), based on the original Gradient-Domain Path Tracing implementation by Kettunen et al. (2015). The scripts used for rendering the high number of training images are based on a simplified version of the batch rendering system from Temporal Gradient-Domain Path Tracing (Manzi et al. 2016). After rendering the images, preferably on a supercomputer, the target images need to be reconstructed and the images need to be combined into datasets.

Reconstructing the target images with the provided 'poisson.exe' requires a CUDA-capable GPU. We recommend TensorFlow version 1.12 or later, compiled with CUDA support.

Creating new datasets requires calling Mitsuba to render a large number of images with different parameters. The provided scene files have been modified so that all important parameters can be given to Mitsuba as commandline options automatically by the provided rendering scripts. As such, the first task is to generate the lists of commandline parameters to be fed to Mitsuba.

### Creating the Render Jobs

To create the render job definitions, go to directory 'create_dataset' and run the script 'run_enqueue_renders.py'.
>cd create_dataset<br>
>python run_enqueue_renders.py

This script creates the required render calls, split into batches 'train1', 'train2', 'train3', 'train4', 'test1', 'test2', and 'test3' by default. You can split the renders differently by modifying 'run_enqueue_renders.py'.

Batches 'train1' - 'train3' contain 128x128 training crops with 2..1024 samples and 'train4' contains their ground truth images. Batches 'test1' and 'test3' contain input images for testing ('test1' with gradients and 'test3' without), and 'test2' contains their ground truth images.

The file 'run_enqueue_renders.py' may be modified e.g. to add more scenes.

### Running the Renders
Next, acquire the scenes used for rendering (see above) if not done already, and place them on directory 'run_on_cluster/run/scenes/' on the platform you will be rendering on, e.g., a Linux or Windows supercomputer.

Let us start by testing the rendering process on Windows with the pre-compiled Mitsuba binary:
>cd run_on_cluster\run<br>
>python run_windows.py train1 train2 train3 train4 test1 test2 test3

This script renders the dataset images on the local computer by running the batches 'train1', ..., 'test3' sequentially. The default core count is 8 and can be changed via the variable CPUS_PER_TASK in 'run_windows.py'. Creating the full dataset will, however, most likely take too long on a single computer. The different batches can rendered on different computers (python run_windows.py BATCH_NAME), but this will likely still take unreasonably long.

The script 'run_windows.py' runs the script 'task_run.py' for the given batches (BATCH_NAME) as ```python task_run.py BATCH_NAME__1 CPU_COUNT TASK_ID``` in sequence for each task (TASK_ID) listed in 'configs/BATCH_NAME__1_parameters.cfg', i.e., it renders the images for the dataset one by one. In practice you will need to run the renders in parallel on a supercomputer or e.g. Amazon AWS. File 'run_windows.py' provides a starting point for the required scripts.

Many universities have access to supercomputers that run Linux and the Slurm Workload Manager. This distribution includes scripts for scheduling the renders with Slurm. See the section *Setting up the Linux + Slurm Environment* below.

The next section assumes that the renders have been successfully rendered, which may take a long time.

### Reconstructing the Target Images
Once the renders have finished, the target images need to be reconstructed unless you configured the targets to use standard Path Tracing with vastly (e.g. 10x or more) more samples.

To reconstruct the target images, go to directory 'create_dataset' and run the script 'run_reconstruct_renders.py'.
>cd create_dataset<br>
>python run_reconstruct_renders.py

For extra performance, the directory "temp_root" in the script could be configured to reside on an SSD drive.

All directories ending in 'run_on_cluster/run/results' that end with '-clean' should now have a new file 'frame000_seed0/image-final.pfm'. If this is not the case, make sure that 'run_reconstruct_renders.py' successfully finds the input images and that the Poisson reconstruction ('bin/poisson.exe') runs correctly. 

The images can now be combined into datasets.

### Combining the Images into Datasets
The datasets can now be reconstructed. Go to the directory 'create_dataset' and run the script 'run_create_datasets.py'.

>cd create_dataset<br>
>python run_create_datasets.py

This will by default create the datasets directly into the 'datasets' directory under the base directory. The network is now ready for training!

IMPORTANT: Remember to take the new datasets into use by modifying the files 'reconstruct/config.py' and 'reconstruct/run_results.py'!


## Setting up the Linux + Slurm Environment

The most complicated step in preparing to run the rendering batches on a supercomputer is compiling Mitsuba on Linux. Once that has been done, Mitsuba needs to be ran in parallel with the supercomputer's workload management system. The following contains some tips for compiling Mitsuba on Linux and running the renders in parallel with the Slurm workload manager.

Start by copying the directory 'run_on_cluster' to your supercomputer, and make sure that the scenes are located in 'run_on_cluster/run/scenes' and directory 'run_on_cluster/run/configs' contains the batch configurations from 'train1' to 'test3'. Directory 'mitsuba' contains a source release of Mitsuba with the up-to-date Gradient-Domain Path Tracing implementation and needs to be compiled first.

### Compiling Mitsuba on Linux
Creating the dataset on a Linux supercomputer requires compiling the provided custom Mitsuba distribution on Linux. This can be relatively tricky and requires familiarity with compiling C++ code on Linux.

You may need to install additional libraries locally, and you may need to edit some config files manually to tell Mitsuba where to find the new libraries. It may be a good idea to start by first making sure that you can compile standard Mitsuba. Some problems will probably arise, and some of the tips below may be of help. Use the official Mitsuba documentation as the primary guide for the compilation.
  
Start by acquiring Scons and any libraries required by Mitsuba. Once you are able to compile the standard Mitsuba (run 'scons' in the source directory), proceed to configuring the provided Gradient-Domain version of Mitsuba:
  
  * Disable the GUI by commenting out the following lines:
      * File 'SConstruct':
          > build('src/mtsgui/SConscript', ['mainEnv', ...
      * File 'build/SConscript.install':
          > if hasQt:<br>
          > &nbsp;&nbsp;&nbsp;&nbsp;install(distDir, ['mtsgui/mtsgui'])
  
  * Copy a suitable configuration script from ‘build/’.
      * Example:
          > cp build/config-linux-gcc.py config.py
  
  * In 'config.py', line 'CXXFLAGS = ...':
      * Add the following flags:
          > '-std=c++11',<br>
          > '-Wno-unused-local-typedefs',<br>
          > '-DDOUBLE_PRECISION'.
      * Remove the following flags:
          > '-DSINGLE_PRECISION',<br>
          > '-DMTS_SSE',<br>
          > '-DMTS_HAS_COHERENT_RT'.
  * In 'config.py', configure and add in any missing INCLUDE and LIBDIR directories.
      * Example:
          > XERCESINCLUDE = ['/path/to/xerces/include']<br>
          > XERCESLIBDIR = ['/path/to/xerces/lib'].
      * Pay extra attention to BOOSTINCLUDE, BOOSTLIBDIR and Boost version. Version 1.54 should work.
      * See Mitsuba documentation for details.
  
  * Compile with 'scons'. Example:
      > scons
  
  * If compilation fails, see file 'config.log'.
  
If all goes well, proceed with the installation by copying the build result from 'dist' to 'run_on_cluster/run/dist'.


### Troubleshooting the Compilation

In case of the following error message while building Mitsuba:
  
  > ‘Could not compile a simple C++ fragment, verify that cl is<br>
  > installed! This could also mean that the Boost libraries are missing. The<br>
  > file "config.log" should contain more information',
  
try bypassing the test in 'build/SConscript.configure' by changing
  
  > 137: conf = Configure(env, custom_tests = { 'CheckCXX' : CheckCXX })
  
  to
  
  > 137: conf = Configure(env).
  
The error messages produced by the compiler now could be more informative.
  
Make sure that the used 'config.py' file uses the DOUBLE_PRECISON flag instead of SINGLE_PRECISION since gradient-rendering in the current form is sensitive to the used precision. This will hopefully be fixed at a later time.
  

### Running the Renders on Linux with Slurm

Ensure that you have compiled Mitsuba on Linux and copied it to 'run_on_cluster/run/dist', have the scenes extracted to 'run_on_cluster/run/scenes', and that the batch configurations are found in 'run_on_on_cluster/run/configs'. Also make sure that a fresh shell is able to run Mitsuba from the command line.

Now you can proceed to running 'task_run.py' for all the batches (BATCH_NAME): train1, train2, train3, train4, test1, test2, test3. The easiest way to do this with Slurm is to use the provided 'run_slurm.py' script:

>python run_slurm.py BATCH_NAME --partition=PARTITION_NAME --cpus=CPU_COUNT --hours=HOURS

You will need to run the command for BATCH_NAMEs from train1 to test3 one by one. The parameter to feed for PARTITION_NAME depends on your Slurm configuration and could be e.g. 'batch' - contact your cluster's administration for help. Parameter HOURS is the time given to each render before Slurm forcefully stops your rendering task and you will need to restart the process manually with more time. It is generally hard to estimate the time required for the renderings and you will most likely need to experiment several times.

The following lists potential rendering times to start experimenting with. Multiply them by 2 or 4 if unsure. This may, however, make the batch system unable to utilize shorter time slots left free from other use, which might delay your results.

| BATCH_NAME |  CPU_COUNT  |  HOURS  |  Description                                |
| ---------- | ----------- | ------- | ------------------------------------------- |
|     train1 |      2      |    1    |  128x128 crops with 2..1024 samples |
|     train2 |      2      |    1    |  128x128 crops with 2..1024 samples |
|     train3 |      2      |    1    |  128x128 crops with 2..1024 samples |
|     train4 |     12      |    4    |  128x128 ground truths with 8192 samples |
|      test1 |     12      |    4    |  1280x720 test images with 2..1024 samples |
|      test2 |     12      |   32    |  1280x720 ground truths with 8192 samples |
|      test3 |     12      |    4    |  1280x720 test images with path tracing, equivalent time |
  
  Table: Recommended per-job rendering times (at least).


To debug a render, run ```python task_run.py BATCH_NAME__1 CPU_COUNT TASK_ID```, e.g., 'python task_run.py train1__1 12 0'. Do this manually on a login node before launching the first batch via 'run_slurm.py' to avoid a potentially failing batch. This makes sure that you can run Mitsuba and load the scenes. Repeat the test on one of the computing nodes. If everything works, first try with a single batch and make sure that it is working well and outputting results to 'run_on_cluster/run/results'. Check the contents of the 'data.zip' files – they should contain meaningful .exr files. If not, something is wrong and the rendering needs to be debugged.

Once the tasks have been rendered, copy the directory 'run/results' back to the local computer at 'run/results' and proceed with creating the datasets as described above in the section 'Reconstructing the Target Images'.


## Full License Information
  
The dataset creation code (directory "create_dataset" with the exception of "create_dataset/pfm.py"),
the neural reconstruction code (directory "reconstruct"), and
the task scripts (run_on_cluster/run/*.py) are covered by the following license:

  Copyright (c) 2016-2019, Markus Kettunen, Erik Härkönen, Jaakko Lehtinen, Aalto University.
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----
  
The file "create_dataset/pfm.py" is from https://gist.github.com/chpatrick/8935738
  
----

E-LPIPS (directory "reconstruct/elpips") is covered by the following license:

  Copyright (c) 2018-2019, Markus Kettunen, Erik Härkönen, Jaakko Lehtinen, Seyoung Park, Aalto University
  Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  
----
  
The screened Poisson reconstruction code (directory "poisson" and file "bin/poisson.exe") is covered by the following license:

  Copyright (c) 2016, Marco Manzi and Markus Kettunen. All rights reserved.
  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
     *  Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     *  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
     *  Neither the name of the NVIDIA CORPORATION nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.


  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----
  
The Mitsuba renderer (directories "mitsuba" and "run_on_cluster/run/dist"), included with this release for convenience, is covered by the following license:

  Copyright (c) 2007-2014 by Wenzel Jakob and others.
  
  Mitsuba is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License Version 3
  as published by the Free Software Foundation.
  
  Mitsuba is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
