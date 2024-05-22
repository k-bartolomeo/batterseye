# batterseye
Classification of players in still images from Baseball Savant clips using 
Siamese networks.

## Introduction
This project involves the classification of players in still images taken from 
broadcasts of MLB gameplay. Datasets are constructed using a combination of the 
MLB Stats API and the Baseball Savant website. Gameplay metadata and tags are 
retrieved using the MLB Stats API, and extracted information is used to construct
URLs to Baseball Savant webpages that have embedded videos of gameplay. Still 
images are retrieved from these embedded videos and paired with the gameplay
metadata to create a supervised learning dataset.

A Siamese network can then be trained on the constructed dataset to identify 
whether pairings of images in the dataset are of the same player or are of 
different players. After crafting image pairs that include an image with a 
known label and an image needing a label, the trained network can then be 
used for classification by generating probabilities that the players in the 
image pairs are the same. For a given image without a label, the known label
from the image in whichever pairing produces the highest probability of 
similarity is assigned to the image with a label. 

## Getting Started 

### Building a Dataset
A dataset can be built using the `scripts/build_dataset.py` script. The script
takes the following parameters at runtime:

- `--team-ids`:  Team IDs for MLB teams. A list of team IDs can be found 
[here](https://github.com/jasonlttl/gameday-api-docs/blob/master/team-information.md).
Default is `147,146,121,111`.
- `--matchup-path`: Path to which matchup metadata is saved. Default is 
`./data/matchups.csv`.
- `--img-dir`: Path to directory to which images are downloaded. Default is
`./data/images`.
- `--img-size`: Height and width in pixels of the images to be downloaded. 
Default is `360`.
- `--threads`: Number of threads to use for multi-threaded downloaded.
Default is `50`.

### Training a Model
A model can be trained after building a dataset as specified in the previous
section. The model can be constructed and then trained using the `scripts/train_model.py`
script. This script takes the following parameters:

- `--img-kind`: Whether to train on pitcher images, batter images, or both. 
Default value is `pitcher`.
- `--img-dir`: Path to directory where images are saved. Default is
`./data/images`.
- `--val-split`: Percentage of dataset to use for validation during training. 
Default value is `0.2`.
- `--test-split`: Percentage of dataset to use for testing. Default value is `0.2`.
- `--train-hash-path`: Path to which hash values for image pairs in training set
should be saved. Default value is `./hashes/train_hashes.txt`.
- `--batch-szie`: Batch size for model training. Default value is `16`.
- `--epochs`: Epochs for model training. Default value is `10`.
- `--input-dim`: Number of channels in images. Default values is `3.`
- `--output-dim`: Output dimensionality of embedding network. Default value is `32`.
- `--layer-type`: Type of convolutional layer to use in embedding network. Default 
value is `residual`.
- `--blocks`: Number of convolutional blocks to use in embedding network. Default 
value is `2`.
- `--filters`: Number of filters to use in convolutional layers. Default value is `32`.
- `--kernel-size`: Kernel size of filters in convolutional layers. Default value is `3`.
- `--strides`: Strides used in convolutional layers. Default value is `1`.
- `--margin`: Margin to use with contrastive loss function. Default value is `1`.
- `--lr`: Learning rate to use for training. Default value is `0.001`.
- `--steps-per-epoch`: Number of steps to taking during each training epoch. Default
value is `20_000`.
- `--validation-steps`: Number of validation steps to taking during each training epoch.
Default value is `5_000`.
- `--weight-path`: Path to which trained model weights should be saved. Default value is
`./weights/siamese.weights.h5'`.

### Evaluating the Trained Model
Once a model has been trained, its multi-class classification accuracy can be evaluated 
using the `scripts/evaluate_classifier.py` script. This script takes the following 
parameters:

- `--matchups`: Path to which matchup metadata is saved. Default is 
`./data/matchups.csv`.
- `--train-hashes`: Path to which hash values for image pairs in training set
are saved. Default value is `./hashes/train_hashes.txt`.
- `--data`: Path to directory where images are saved. Default is
`./data/images`.
- `--img-kind`: Whether to model was trained on pitcher images, batter images,
or both. Default value is `pitcher`.
- `--batch-szie`: Batch size used for training. Default value is `16`.
- `--input-dim`: Number of channels in images. Default values is `3.`
- `--output-dim`: Output dimensionality of embedding network. Default value is `32`.
- `--layer-type`: Type of convolutional layer used in embedding network. Default 
value is `residual`.
- `--blocks`: Number of convolutional blocks used in embedding network. Default 
value is `2`.
- `--filters`: Number of filters used in convolutional layers. Default value is `32`.
- `--kernel-size`: Kernel size of filters in convolutional layers. Default value is `3`.
- `--strides`: Strides used in convolutional layers. Default value is `1`.
- `--weights`: Path at which trained model weights are saved. Default value is
`./weights/siamese.weights.h5'`.
