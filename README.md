# gender classifier

A CNN for classifying gender from gait, using TensorFlow.

The main focus is to look at gait representations of various lengths and determine the lowest amount of data necessary to use gait as a feature.
The representations being analysed are GEI, sub-GEI, Local frame average, Key frames and single frame.

Determining gender is not the main focus, just an easier/quicker gait-based classification problem for training and testing on.
When all representations have been tested the best performing will be trained on a full gait recognition dataset.

## Installation

Install Ananconda

To install the env:
```
conda env create -f environment.yml
```

To activate the env: 
```
conda activate gender-classifier
```

## Testing

Use the following to launch the testing UI:
```
python launch_ui.py
```

### To align silhouettes
Setup cython utils:
```
cd utils

python setup.py build_ext --inplace

python align.py $SILHOUETTE_DIRECTORY
```

## Progress

Classifiers have been trained for the following representations:

-GEI.

-Single frame.

-Key frames.

A basic UI has been developed to test individual frames on each model.

