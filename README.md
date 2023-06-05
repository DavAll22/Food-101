# Food-101

Project using the Food-101 dataset to classify multiple image classes using the EfficientNetB0 CNN arctitecture (https://arxiv.org/pdf/1905.11946.pdf) built in TensorFlow 2.X.

## This project uses:
* TensorFlow 2.X to create, train and evaluate the performance of a multi-class image classification dataset
* Functional API to utilise the EfficientNetB0 network as the backbone of the network, using transfer learning to adapt the model to the dataset
* Image manipulation using data augmentation to help prevent overfitting
* Dataset batching, Pre-fetching and mixed precision training to improve model training time
* Callbacks in fine-tuning to reduce the model learning rate with `EarlyStopping()` and `ReduceLROnPlateau()`

Food-101 dataset contains images of 101 classes of food, with 101k images split into 75,750 training sample and 25,250 test samples (75:25). The images have a maximum width of 512 pixels and are re-scaled to be 224x224 pixels for the classifier model. The model validates using the test data during the training of the model.

The labels used in the dataset are not one-hot encoded and so in training, `sparse_categorical_crossentropy` is used in the loss function

The model uses transfer learning with EfficientNetB0 as the backbone classifier, being trained on ImageNet. The weights of this network are frozen and are untrainable, so retains the information it learned from classifying on ImageNet.
The top of EfficientNetB0 was not included to allow adding a custom output layer for the Food-101 dataset, which are unfrozen and will be trained by the model.
![image](https://github.com/DavAll22/Food-101/assets/124359152/df1c7948-f734-4b7b-a0c7-a3a997549381)

The trained model was cloned for fine-tuning using a 10x lower learning rate with callbacks to reduce the learning rate every epoch until a set limit.

## Evaluation
* On the test data, the model achieves a loss of 0.859 and an accuracy of 80.26%
  * The loss indicates the model is overfitting and further fine-tuning is required (e.g. more data augmentation methods applied)
 
![image](https://github.com/DavAll22/Food-101/assets/124359152/31687f10-7eab-4e9b-aa6f-7332470b928c)


## Predictions
* Prediction probabilities for each class is generated from the test data and the corresponding class label is assigned to the maximum prediction from the model
* A confusion matrix is plotted from the predictions
  * The model gets confused over classes which have similar features, like *Filet_Mignon* and *Steak*

The per-class F1-Score is evaluated and shown that class *Steak* is the worst performing from the 101 classes. More data could be obtained for this class to improve the score.
![image](https://github.com/DavAll22/Food-101/assets/124359152/a04510a0-603b-4f1b-97dc-8a9ceada4796)
