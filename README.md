# ML_Project27-TrafficSignRecognition

### Traffic Sign Classification with TensorFlow and Keras
This project implements a Convolutional Neural Network (CNN) to classify traffic signs using TensorFlow and Keras. It aims to achieve high accuracy in identifying various traffic signs from images.

### Getting Started
##### This project requires the following libraries:
```
TensorFlow
Keras
NumPy
Pillow (PIL Fork)
scikit-learn (optional, for data splitting)
matplotlib (optional, for data visualization)
```

### You can install them using pip:
```
pip install tensorflow keras numpy Pillow scikit-learn matplotlib
```


### Data
The project expects a dataset of traffic sign images organized in folders by class label. Each folder should contain images representing that specific traffic sign.

Alternatively, you can explore using publicly available datasets like the German Traffic Sign Recognition Benchmark (GTSRB) https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

### Usage
1.Preprocess the data: Run the data_preprocess.py script to preprocess the images. This script resizes images, normalizes pixel values, and converts them to NumPy arrays with corresponding class labels. The script saves the preprocessed data for future use.

2.Train the model: Run the train_model.py script to train the CNN model. This script loads the preprocessed data, splits it into training and testing sets, builds the CNN model, compiles it, and trains it for a specified number of epochs.

3.Evaluate the model: After training, the script evaluates the model's performance on the testing set. You can also use the trained model to make predictions on new traffic sign images.

### Project Structure
```
traffic_sign_classification/
├── data/               # Folder to store your traffic sign dataset
│   └── class1/         # Folder containing images of class 1 signs
│       └── ...          # More class folders
├── data_preprocess.py  # Script for preprocessing data
├── train_model.py       # Script for training the CNN model
└── README.md            # This file (instructions)
```

### Notes
This is a basic implementation of a CNN for traffic sign classification. You can experiment with different hyperparameters (number of filters, kernel size, etc.) and model architectures to improve performance.

Consider using techniques like data augmentation to increase the size and diversity of your training data.

This project provides a starting point for building your own traffic sign classification system. Feel free to modify and extend the code to fit your specific needs.
