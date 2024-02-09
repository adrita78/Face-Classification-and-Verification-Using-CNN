# Face Classification and Verification
Implemented CNN-based architectures like ConvNeXt to pattern recognition problems that require position invariance. Worked on the problem of recognizing or verifying faces in images. 

# Problem Setting

In this project, we address two kinds of problems:

1. **Classification Problem**: Our goal is to identify the person in a given picture. This is a closed-set problem, where the subjects in the test set are also present in the training set, although the specific pictures in the test set may not be included in the training data. Achieving high accuracy in this task requires that the embeddings for all subjects in our "vocabulary" are linearly separable from each other.

2. **Verification Problem**: Here, our objective is to determine if the person in a query picture is also present in a given gallery of images. This is an open-set problem, where the subjects in the test data may not have been seen during training. To solve this problem effectively, we need to ensure that the embeddings of two pictures of the same person are always closer to each other than the embeddings of pictures of two different people. Thus, the goal is to determine if the embedding of any picture in the gallery is sufficiently close to that of the query.


## Dependencies

Make sure you have the following dependencies installed:

1. Python 3.6+
2. PyTorch 2.0
3. Numpy
4. Matplotlib
5. Wandb
6. DataLoader, TensorDataset

## Running the Code

Once you have installed all the dependencies and downloaded the dataset, you can run the code by opening the notebook in your Google Colab environment.

## Notebook Sections

The notebook is structured into the following sections:

1. Data Loading
2. Classification Test Dataset Class
3. Model Architecture (Tiny ConvNeXt)
4. Hyperparameter Tuning
5. Training the Model
6. Evaluating the Model
7. Hyperparameter Tuning
8. Testing the Classification Model
9. Verification Task
10. Evaluating Verification
11. Submitting Results to Kaggle

## Experiments

### Model Architecture

Different architectures and hyperparameters were experimented with to achieve the best performance. ResNet34 and ResNet50 were initially tried but did not meet the cutoff. The Tiny ConvNeXt model gave the best performance, achieving an accuracy of 91.2420% on the classification task.

### Number of Epochs

The model was trained for 230 epochs, with performance plateauing after 200 epochs.

### Hyperparameters

The following hyperparameters were used:

- Learning Rate: 0.01 (Reduced to 0.001 after 100 epochs)
- Batch Size: 64
- Label Smoothing: 0.2
- Criterion: CrossEntropyLoss
- Optimizer: SGD
- Scheduler: CosineAnnealingLR

### Data Loading Scheme

PyTorch's DataLoader was used to load the data. Data augmentation techniques like random horizontal flip, random grayscale, Rand Augment, and ColorJitter were applied to improve the model's performance.

### Verification Task

For the verification task, a threshold value of 0.3 gave the best results. Attempts to improve the classification model's performance using advanced losses like CenterLoss and Arc Face loss did not yield significant improvement. Instead, the classification model was trained for more epochs to achieve better verification accuracy, reaching 64.7% (61.1% on Kaggle).
