## Detecting Presence of Brain Aneurysms - A Deep Learning Approach

We develop a pipeline for detecting the presence and arterial location of brain aneurysms from three-dimensional brain scans. This repository contains three files. The first file, preprocessing.ipynb, preprocesses the brain scans and prepares them for analysis. The second, one-step.ipynb, employs a 3D Convolutional Neural Network (CNN) to detect the presence and location in a united approach. The third, two-step.ipynb, implements 2 3D CNNs. The first 3D CNN detects the presence of an aneurysm and the second 3D CNN focuses on its location. In the steps below, we detail the procedure for implementing this method.

## Download the Data from Kaggle
The brain scan data is too large to store on GitHub, so you must download the [data](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data) from the Kaggle Competition.

## Pre-Process the Data
Both the one-step and two-step models use the same preprocessing; they differ in how detection and localization are learned (coupled vs. decoupled).  
All data is stored in HDF5 shard files for memory-efficient training.

For each subject, the preprocessing step:
- loads all slices in the scan,
- orders slices using DICOM geometry,
- applies intensity normalization and windowing,
- resizes slices to a uniform grid,
- writes the preprocessed slices to an HDF5 shard.

To cover the full dataset, run preprocessing.ipynb four times, each time with a different shard index (for example, 0, 1, 2, and 3).  
Mini-volumes (32 slices) are **not** created in this step; they are created later in the one-step.ipynb or two-step.ipynb .

### How to Run
1. Run preprocessing.ipynb four times, each time with a different shard index, to create the HDF5 shards.
2. Choose a training pipeline:
   - one_step.ipynb for the coupled model, or
   - two_step.ipynb for the decoupled model.
3. Point the chosen training notebook to the directory containing the shards.
4. Run the notebook to train the model; mini-volume construction and label assignment are handled internally.

## One Step Model
Once you have downloaded the pre-processed data using preprocessing.ipynb, be sure to adjust the file path to the shards in one-step.ipynb (code chunk 4). From here, the file will divide the pre-processed brains scans into mini-volumes and train, validation, and test sets. We use a 70-10-20 split. The dataloaders will load the brain scans while adhering to memory constraints. Then, we initiliaze a 3D CNN with three convolutional blocks and three feedforward layers. The model is trained on the training data for 5 epochs using a weighted cross entropy loss. We print the training and validation loss curves and save the model at the epoch with the lowest validation loss. Next, the trained model is applied to the test set and the weighted ROC AUC is printed for evaluation. Often, the model is applied to multiple minivolumes corresponding to the same subject. In these cases, we apply a simple decision rule to aggregate the predictions to yield a single output for each test subject. In this model, we achieve a weighted ROC AUC of 0.6249. The model provides helpful inference into the presence of an aneurysm, but is unable to identify the location of the aneurysm.

## Two Step Model
The Two Step model is two-step.ipynb is similar. Again, be sure to adjust the file path to the shards (code chunk 4). Pre-processed brains scans are converted into mini-volumes and split into train, validation, and test sets with a 70-10-20 split. The two-step approach utilizes two 3D CNN; the major difference between them being the classification head. The first CNN identifies the presence of an aneurysm, while the second CNN predicts the arterial location of the aneurysm. The first model is trained on the entire training data for 6 epochs using a binary cross entropy loss, while the second model is trained on the subset containing aneurysms for 10 epochs. After training the first model, the weights for the second CNN are initialized using the first CNN, and the first five convolutional layers are frozen during training. We print the training and validation loss curves for both models and save the model at the epoch with the lowest validation loss. In the evaluation step, the data is applied to the first model. If an aneurysm is predicted, the data is then applied to the second model. In the two-step model, we achieve a weighted ROC AUC of 0.6224. Despite our best efforts, the model is still unable to meaninful predict the arterial location of the aneurysm.

## Code References
https://medium.com/data-science/hdf5-datasets-for-pytorch-631ff1d750f5
https://github.com/pytorch/pytorch/issues/11929

