# DOGGO-CLASSIFIER🐕

## Multi-class Dog Breed Classification using Convolutional Neural Networks (CNN)
This project is a multi-class classification task to identify the breed of a dog from an image using Convolutional Neural Networks (CNN). The dataset used for this task is the Stanford Dogs Dataset, which contains 20,580 images of 120 dog breeds.

**The project is implemented in Python 3.7 using the following libraries:**
- TensorFlow 2.5.0
- Keras 2.4.3
- NumPy 1.19.5
- Matplotlib 3.3.4
- Scikit-learn 0.24.2

**The project consists of the following files:**
- Dog_vision.ipynb: Jupyter Notebook containing the implementation of the CNN model and training process.
- model.png: Image of the CNN model architecture.
- requirements.txt: File containing the required libraries for running the project.

## Installation 📦
To install the required libraries, run the following command:
pip install -r requirements.txt

## Usage 👩‍🎨
To run the project, open the Dog_vision.ipynb notebook and run the cells. The notebook contains step-by-step instructions on how to download and preprocess the dataset, build and train the CNN model, and evaluate the performance of the model.

## Results 💯
The trained model achieved an accuracy of 80.83% on the test set. The model was able to correctly classify the breed of the dog in most cases, but struggled with some breeds that look similar. The model can be further improved by using more advanced techniques such as data augmentation, transfer learning, and ensembling.

## Credits 💌
The implementation of the CNN model is based on the following resources:
[TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
[Convolutional Neural Networks (CNNs) for Image Classification](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome) by Andrew Ng on Coursera.
