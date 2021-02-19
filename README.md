DCASE2021 - Task 1 A - Baseline systems
-------------------------------------

Author:
**Irene Martin**, *Tampere University* 
[Email](mailto:irene.martinmorato@tuni.fi), 
Adaptations from the original code DCASE2020 - Task 1 by
**Toni Heittola**, *Tampere University* 


Getting started
===============

1. Clone repository from [Github](https://github.com/marmoi/dcase2021_task1a_baseline).
2. Install requirements with command: `pip install -r requirements.txt`.
3. Run the task specific application with default settings for two model quantization:
   - keras model: `python task1a.py` or  `./task1a.py`
   - TFLite: `python task1a_tflite.py` or  `./task1a_tflite.py`

### Anaconda installation

To setup Anaconda environment for the system use following:

	conda create --name tf2-dcase python=3.6
	conda activate tf2-dcase
	conda install ipython
	conda install numpy
	conda install tensorflow-gpu=2.1.0
	conda install -c anaconda cudatoolkit
	conda install -c anaconda cudnn
	pip install librosa
	pip install absl-py==0.9.0
	pip install dcase_util==0.2.12
	pip install sed_eval
	pip install pyyaml==5.3.1
	
Introduction
============

This is the baseline system for the [Acoustic scene classification task (Task 1A)] in Detection and Classification of Acoustic Scenes and Events 2021 (DCASE2021) challenge. The system is intended to provide a simple entry-level state-of-the-art approach that gives reasonable results in the subtask 1A. The baseline system is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox (>=version 0.2.16). 

Participants can build their own systems by extending the provided baseline system. The system has all needed functionality for the dataset handling, acoustic feature storing and accessing, acoustic model training and storing, and evaluation. The modular structure of the system enables participants to modify the system to their needs. The baseline system is a good starting point especially for the entry level researchers to familiarize themselves with the acoustic scene classification problem. 

If participants plan to publish their code to the DCASE community after the challenge, building their approach on the baseline system could potentially make their code more accessible to the community. DCASE organizers strongly encourage participants to share their code in any form after the challenge.

Description
========

### Subtask A - Low-Complexity Acoustic Scene Classification with Multiple Devices

[TAU Urban Acoustic Scenes 2020 Mobile Development dataset](https://zenodo.org/record/3819968) is used as development dataset for this task.

This subtask is concerned with the basic problem of acoustic scene classification, in which it is required to classify a test audio recording into one of ten known acoustic scene classes. This task targets **generalization** properties of systems across a number of different devices, and will use audio data recorded and simulated with a variety of devices. 
Recordings in the dataset were made with three devices (A, B and C) that captured audio simultaneously and 6 simulated devices (S1-S6). Each acoustic scene has 1440 segments (240 minutes of audio) recorded with device A (main device) and 108 segments of parallel audio (18 minutes) each recorded with devices B,C, and S1-S6. The dataset contains in total 64 hours of audio. For a more detailed description see [DCASE Challenge task description](http://dcase.community/challenge2020/task-acoustic-scene-classification).

The task targets low complexity solutions for the classification problem in term of model size, and uses audio recorded with a single device (device A, 48 kHz / 24bit / stereo). The data for the dataset was recorded in 10 acoustic scenes which were later grouped into three major classes used in this subtask. The dataset contains in total 40 hours of audio. For a more detailed description see [DCASE Challenge task description](http://dcase.community/challenge2020/task-acoustic-scene-classification).

Classifier complexity for this subtask is limited to 128KB size for the non-zero parameters. This translates into 32K parameters when using float32 (32-bit float) which is often the default data type (32000 parameter values * 32 bits per parameter / 8 bits per byte= 128000 bytes = 128KB). See detailed description how to calculate model size from [DCASE Challenge task description](http://dcase.community/challenge2020/task-acoustic-scene-classification). Model calculation for Keras models is implemented in `model_size_calculation.py`

The subtask specific baseline system is implemented in file `task1a.py` and `task1a_tflite.py`.

#### System description

The system implements a convolutional neural network (CNN) based approach, where log mel-band energies are first extracted for each 10-second signal, and a network consisting of two CNN layers and one fully connected layer is trained to assign scene labels to the audio signals. Model size of the baseline is 90.82 KB.


##### Parameters

###### Acoustic features

- Analysis frame 40 ms (50% hop size)
- Log mel-band energies (40 bands)

###### Neural network

- Input shape: 40 * 500 (10 seconds)
- Architecture:
  - CNN layer #1
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (5, 5)) 
  - CNN layer #2
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (5, 5)) + Dropout (rate: 30%)
  - CNN layer #2
    - 2D Convolutional layer (filters: 32, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (4, 100)) + Dropout (rate: 30%)
  - Flatten
  - Dense layer #1
    - Dense layer (units: 100, activation: ReLu )
    - Dropout (rate: 30%)
  - Output layer (activation: softmax/sigmoid)
- Learning (epochs: 200, batch size: 16, data shuffling between epochs)
  - Optimizer: Adam (learning rate: 0.001)
- Model selection:
  - Approximately 30% of the original training data is assigned to validation set, split done so that training and validation sets do not have segments from same location and so that both sets have similar amount of data per city
  - Model performance after each epoch is evaluated on the validation set, and best performing model is selected
  
**Network summary**
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	conv2d (Conv2D)              (None, 40, 500, 16)       800       
	_________________________________________________________________
	batch_normalization (BatchNo (None, 40, 500, 16)       64        
	_________________________________________________________________
	activation (Activation)      (None, 40, 500, 16)       0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 40, 500, 16)       12560     
	_________________________________________________________________
	batch_normalization_1 (Batch (None, 40, 500, 16)       64        
	_________________________________________________________________
	activation_1 (Activation)    (None, 40, 500, 16)       0         
	_________________________________________________________________
	max_pooling2d (MaxPooling2D) (None, 8, 100, 16)        0         
	_________________________________________________________________
	dropout (Dropout)            (None, 8, 100, 16)        0         
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 8, 100, 32)        25120     
	_________________________________________________________________
	batch_normalization_2 (Batch (None, 8, 100, 32)        128       
	_________________________________________________________________
	activation_2 (Activation)    (None, 8, 100, 32)        0         
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 2, 1, 32)          0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 2, 1, 32)          0         
	_________________________________________________________________
	flatten (Flatten)            (None, 64)                0         
	_________________________________________________________________
	dense (Dense)                (None, 100)               6500      
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 100)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 10)                1010      
	=================================================================
	Total params: 46,246
	Trainable params: 46,118
	Non-trainable params: 128
	_________________________________________________________________

     Input shape                     : (None, 40, 500, 1)
     Output shape                    : (None, 10)

  
#### Results for development dataset

The cross-validation setup provided with the *TAU Urban Acoustic Scenes 2020 Mobile Development dataset* is used to evaluate the performance of the baseline system. Results are calculated using TensorFlow in GPU mode (using Nvidia Tesla V100 GPU card). Because results produced with GPU card are generally non-deterministic, the system was trained and tested 10 times, and mean and standard deviation of the performance from these 10 independent trials are shown in the results tables.

| Scene class   | Accuracy |   A       B       C       S1      S2      S3      S4      S5      S6  | Logloss  
| ------------- | ------   | -----   -----   -----   -----   -----   -----   -----   -----   ----- | -------  
| airport       | 26.0%    | 45.5    15.6    33.3    12.1    36.4    33.3    12.1    24.2    21.2  | 1.527    
| bus           | 38.7%    | 72.7    30.3    60.6    21.2    45.5    39.4    21.2    15.2    42.4  | 1.689    
| metro         | 44.4%    | 60.6    66.7    54.5    18.2    30.3    33.3    45.5    39.4    51.5  | 1.633    
| metro_station | 39.7%    | 48.5    45.5    30.3    30.3    33.3    48.5    30.3    42.4    48.5  | 1.708    
| park          | 62.3%    | 90.9    87.9    93.9    72.7    57.6    63.6    24.2    39.4    30.3  | 1.333    
| public_square | 34.0%    | 51.5    33.3    30.3    42.4    21.2    54.5    33.3    30.3    9.1   | 1.864    
| shopping_mall | 63.3%    | 63.6    84.8    69.7    45.5    60.6    63.6    66.7    33.3    81.8  | 1.102    
| street_pede.. | 43.4%    | 51.5    51.5    51.5    54.5    57.6    48.5    21.2    9.1     45.5  | 1.644    
| street_traf.. | 60.6%    | 72.7    63.6    63.6    72.7    57.6    54.5    57.6    72.7    30.3  | 1.396    
| tram          | 40.9%    | 66.7    33.3    62.5    57.6    12.1    33.3    27.3    45.5    30.3  | 1.610    
| ------------- | ------   | -----   -----   -----   -----   -----   -----   -----   -----   ----- | -------  
| Accuracy      | 45.3%    | 62.4    51.3    55.0    42.7    41.2    47.3    33.9    35.2    39.1  |          
| Logloss       |          | 1.003   1.325   1.256   1.591   1.661   1.387   1.978   1.912   1.840 | 1.551    



Usage
=====

For the subtask there are two separate application (.py file):

- `task1a.py`, DCASE2021 baseline for Task 1A, with keras model quantization
- `task1a_tflite.py`, DCASE2021 baseline for Task 1A, with TFLite quantification

### Application arguments

All the usage arguments are shown by ``python task1a.py -h``.

| Argument                    |                                   | Description                                                  |
| --------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `-h`                        | `--help`                          | Application help.                                            |
| `-v`                        | `--version`                       | Show application version.                                    |
| `-m {dev,eval}`             | `--mode {dev,eval}`               | Selector for application operation mode                      |
| `-s PARAMETER_SET`          | `--parameter_set PARAMETER_SET`   | Parameter set id. Can be also comma separated list e.g. `-s set1,set2,set3``. In this case, each set is run separately. |
| `-p FILE`                   | `--param_file FILE`               | Parameter file (YAML) to overwrite the default parameters    |
| `-o OUTPUT_FILE`            | `--output OUTPUT_FILE`            | Output file                                                  |
|                             | `--overwrite`                     | Force overwrite mode.                                        |
|                             | `--download_dataset DATASET_PATH` | Download dataset to given path and exit                      |
|                             | `--show_parameters`               | Show active application parameter set                        |
|                             | `--show_sets`                     | List of available parameter sets                             |
|                             | `--show_results`                  | Show results of the evaluated system setups                  |

### Operation modes

The system can be used in three different operation modes.

**Development mode** - `dev`

In development mode, the development dataset is used with the provided cross-validation setup: training set is used for learning, and testing set is used for evaluating the performance of the system. This is the default operation mode. 

Usage example: `python task1a.py` or `python task1a.py -m dev`

**Challenge mode** - `eval` 

**Note:** This operation mode does not work yet as the evaluation dataset has not been published. 

In challenge mode, the full development dataset (including training and test subsets) is used for learning, and a second dataset, evaluation dataset, is used for testing. The system system outputs are generated based on the evaluation dataset. If ground truth is available for the evaluation dataset, the output is also evaluated. This mode is designed to be used for generating the DCASE challenge submission, running the system on the evaluation dataset for generating the system outputs for the submission file. 

Usage example: `python task1a.py -m eval` and `python task1b.py -m eval`

To save system output to a file: `python task1a.py -m eval -o output.csv`

### System parameters

The baseline system supports multi-level parameter overwriting, to enable flexible switching between different system setups. Parameter changes are tracked with hashes calculated from parameter sections. These parameter hashes are used in the storage file paths when saving data (features/embeddings, model, or results). By using this approach, the system will compute features/embeddings, models and results only once for the specific parameter set, and after that it will reuse this precomputed data.

#### Parameter overwriting

Parameters are stored in YAML-formatted files, which are handled internally in the system as Dict like objects (`dcase_util.containers.DCASEAppParameterContainer`). **Default parameters** is the set of all possible parameters recognized by the system. **Parameter set** is a smaller set of parameters used to overwrite values of the default parameters. This can be used to select methods for processing, or tune parameters.

#### Parameter file

Parameters files are YAML-formatted files, containing the following three blocks:

- `active_set`, default parameter set id
- `sets`, list of dictionaries
- `defaults`, dictionary containing default parameters which are overwritten by the `sets[active_set]`

At the top level of the parameter dictionary there are parameter sections; depending on the name of the section, the parameters inside it are processed sometimes differently. Usually there is a main section (`feature_extractor`, and method parameter section (`feature_extractor_method_parameters`) which contains parameters for each possible method. When parameters are processed, the correct method parameters are copied from method parameter section to the main section under parameters. This allows having many methods ready parametrized and easily accessible.

#### Parameter hash

Parameter hashes are MD5 hashes calculated for each parameter section. In order to make these hashes more robust, some pre-processing is applied before hash calculation:

- If section contains field `enable` with value `False`, all fields inside this section are excluded from the parameter hash calculation. This will avoid recalculating the hash if the section is not used but some of these unused parameters are changed.
- If section contains fields with value `False`, these fields are excluded from the parameter hash calculation. This will enable to add new flag parameters without changing the hash. Define the new flag such that the previous behaviour is happening when this field is set to false.
- All `non_hashable_fields` fields are excluded from the parameter hash calculation. These fields are set when `dcase_util.containers.AppParameterContainer` is constructed, and they usually are fields used to print various values to the console. These fields do not change the system output to be saved onto disk, and hence they are excluded from hash.


## Extending the baseline

Easiest way to extend the baseline system is by modifying system parameters. To do so one needs to create a parameter file with a custom parameter set, and run system with this parameter file.

**Example 1**

In this example, one creates MLP based system. Data processing chain is replaced with a chain which calculated mean over 500 feature vectors. Learner is replaced with a new model definition. Parameter file `extra.yaml`: 
        
    active_set: minimal-mlp
    sets:
      - set_id: minimal-mlp
        description: Minimal MLP system
        data_processing_chain:
          method: mean_aggregation_chain
        data_processing_chain_method_parameters:
          mean_aggregation_chain:
            chain:
              - processor_name: dcase_util.processors.FeatureReadingProcessor
              - processor_name: dcase_util.processors.NormalizationProcessor
                init_parameters:
                  enable: true
              - processor_name: dcase_util.processors.AggregationProcessor
                init_parameters:
                  aggregation_recipe:
                    - mean
                  win_length_frames: 500
                  hop_length_frames: 500
              - processor_name: dcase_util.processors.DataShapingProcessor
                init_parameters:
                  axis_list:
                    - time_axis
                    - data_axis
        learner:
          method: mlp_mini
        learner_method_parameters:
          mlp_mini:
            random_seed: 0
            keras_profile: deterministic
            backend: tensorflow
            validation_set:
              validation_amount: 0.20
              balancing_mode: identifier_two_level_hierarchy
              seed: 0
            data:
              data_format: channels_last
              target_format: same
            model:
              config:
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                    input_shape:
                      - FEATURE_VECTOR_LENGTH
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax
            compile:
              loss: categorical_crossentropy
              metrics:
                - categorical_accuracy
            optimizer:
              class_name: Adam
            fit:
              epochs: 50
              batch_size: 64
              shuffle: true
            callbacks:
              StasherCallback:
                monitor: val_categorical_accuracy
                initial_delay: 25

Command to run the system:

    python task1a.py -p extra.yaml

**Example 2**

In this example, one slightly modifies the baseline to have smaller network. Learner is replaced with modified model definition. Since `cnn` learner method is overloaded, only a subset of the parameters needs to be defined. However, the model config (network definition) has to be redefined fully as list parameters cannot be overloaded partly. Parameter file `extra.yaml`: 
        
    active_set: baseline-minified
    sets:
      - set_id: baseline-minified
        description: Minified DCASE2021 baseline subtask A minified
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 3            
        
              config:
                - class_name: Conv2D
                  config:
                    input_shape:
                      - FEATURE_VECTOR_LENGTH   # data_axis
                      - INPUT_SEQUENCE_LENGTH   # time_axis
                      - 1                       # sequence_axis
                    filters: 8
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 5
                      - 5
                    data_format: DATA_FORMAT
                - class_name: Conv2D
                  config:
                    filters: 16
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 4
                      - 100
                    data_format: DATA_FORMAT
                - class_name: Flatten      
                - class_name: Dense
                  config:
                    units: 100
                    kernel_initializer: uniform
                    activation: relu    
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax                        
            fit:
                epochs: 100
                                  
Command to run the system:

    python task1a.py -p extra.yaml


**Example 3**

In this example, multiple different setups are run in a sequence. Parameter file `extra.yaml`: 
        
    active_set: baseline-kernel3
    sets:
      - set_id: baseline-kernel3
        description: DCASE2021 baseline for subtask A with kernel 3
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 3
            fit:
              epochs: 100                    
      - set_id: baseline-kernel5
        description: DCASE2021 baseline for subtask A with kernel 5
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 5
            fit:
              epochs: 100
                
Command to run the system:

    python task1a.py -p extra.yaml -s baseline-kernel3,baseline-kernel5

To see results:
    
    python task1a.py --show_results

Code
====

The code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox, see [manual for tutorials](https://dcase-repo.github.io/dcase_util/index.html). The machine learning part of the code in built on [Keras (v2.2.2)](https://keras.io/), using [TensorFlow (v1.14.0)](https://www.tensorflow.org/) as backend.

### File structure

      .
      ├── task1a.py                     # Baseline system for subtask A
      ├── task1a.yaml                   # Configuration file for task1a.py
      ├── task1a_tflite.py              # Baseline system for subtask A with TFLite quantification
      |
      ├── model_size_calculation.py     # Utility function for calculating model size 
      ├── utils.py                      # Common functions shared between tasks
      |
      ├── README.md                     # This file
      └── requirements.txt              # External module dependencies

Changelog
=========

#### 2.0.0 / 2021-02-19


License
=======

This software is released under the terms of the [MIT License](https://github.com/toni-heittola/dcase2020_task1_baseline/blob/master/LICENSE).
