# gender classifier

A CNN for classifying gender using gait.

The main focus is to look at gait representations of various lengths and determine the lowest amount of data necessary to use gait as a feature.
The representations being analysed are GEI, sub-GEI, Local frame average, Key frames and single frame.

Determining gender is not the main focus, just an easier/quicker gait-based classification problem for training and testing on.
When all representations have been tested the best performing will be trained on a full gait recognition dataset.

## Progress

Classifiers have been trained for the following representations:

GEI.
Single frame.

A basic UI has been developed to test individual frames on each model.

