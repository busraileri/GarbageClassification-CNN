# Garbage Classification - CNN

This project involves the development of a Deep Learning model from scratch to classify waste types. 
The aim is to categorize waste images into six different classes using a Convolutional Neural Network (CNN) model. 
Throughout the project, steps such as data loading, data preprocessing, modeling, and model training were applied.

## Project Structure
- **Data Loading**: The dataset, sourced from [data](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data), is loaded and split into training and validation sets.
- **Data Preprocessing**: Images are resized, normalized, and augmented to improve model generalization.
- **Model Architecture**: A CNN model with several convolutional, pooling, and dense layers was built from scratch.
- **Training**: The model is trained over 100 epochs, with early stopping and best model checkpoints.
- **Evaluation**: Model performance is assessed through accuracy, precision, and recall metrics.
- **Test**: It tests the accuracy of the waste classification model by making predictions on 4 different samples and visualizing the results.

### Requirements
- Tensorflow
- Keras
- Matplotlib
- Numpy
- Sklearn 
- Warnings


## 1. **Data Overview**
The dataset used is **TrashNet**, which consists of **2527 images** divided into six categories:
- Glass: 501 images
- Paper: 594 images
- Cardboard: 403 images
- Plastic: 482 images
- Metal: 410 images
- Trash: 137 images

The images have a resolution of **512 x 384 pixels** and 3 RGB channels.

## 2. **Data Preprocessing**
- **Loading Dataset**: OpenCV (`cv2.imread`) is used to read and resize images to a target size of **224x224 pixels**.
- **Shuffling**: The dataset is shuffled to ensure randomness.
- **Image Labels**: Labels are created by extracting the folder name from each image's path and mapping them to numerical values using the `waste_labels` dictionary.

## 3. **Data Visualization**
A function is created to visualize a sample of the dataset using **Matplotlib**. This helps in understanding the variety of images and their corresponding labels.

## 4. **Data Augmentation**
Several augmentations are applied using `ImageDataGenerator`:
- **Training Generator**: Applies random transformations like horizontal and vertical flips, zoom, shear, and shifts to augment the training data.
- **Testing Generator**: Only rescales the images to normalize pixel values.

## 5. **CNN Model Architecture**
The CNN model is built with the following layers:
- **Convolutional Layers**: Three `Conv2D` layers with ReLU activation for feature extraction.
- **MaxPooling Layers**: Used after each convolutional layer to reduce dimensionality.
- **Flatten Layer**: Flattens the output from the convolutional layers.
- **Dense Layers**: Fully connected layers with dropout for regularization, reducing overfitting.
- **Output Layer**: A softmax activation layer with 6 units (one for each class).

The model summary indicates a total of **1,645,830 trainable parameters**.

## 6. **Model Compilation**
The model is compiled using:
- **Categorical cross-entropy** loss
- **Adam optimizer**
- Metrics: **Accuracy**, **Precision**, and **Recall**

## 7. **Model Training**
- The model is trained for **100 epochs** with a batch size of **32**.
- The training performance improves over epochs, with the model reaching an accuracy of **75.2%** and validation accuracy around **65.9%** by the end of training.
- During training, metrics such as **training accuracy**, **validation accuracy**, **loss**, **precision**, and **recall** are monitored.

## 8. **Model Evaluation**
The model's performance has shown steady improvement, although the validation accuracy and loss suggest there may still be room for improvement in generalization.

## Next Steps:
- **Model Fine-Tuning**: Experiment with fine-tuning the model, adjusting hyperparameters (e.g., learning rate, batch size), or adding more convolutional layers.
- **Class Imbalance**: The "trash" category has fewer samples, which could cause the model to underperform on this class. Techniques like **class weighting** or **oversampling the minority class** could be considered.
- **Evaluation**: After training, evaluate the model further using **confusion matrices** or **classification reports** to gain deeper insights into performance across different classes.
