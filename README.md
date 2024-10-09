# Custom-Face-Detection-with-CNNs
This project implements a custom face detection system from scratch using Convolutional Neural Networks (CNNs). The system is designed to detect human faces in images, offering flexibility in training with custom datasets.
## Approach

The face detection system consists of the following steps:

1. **Data Loading**: Images are loaded from two datasets consisting faces and non-faces. Each image is resized to 64x64 pixels.
 - **Dataset 1: faces_dataset**
     - This dataset consists only of faces of humans and is labeled as 1.
 - **Dataset 2: non_faces_dataset**
     - This dataset consists of images of flowers, airplanes, motorcycles, nature, dogs, and cats, which are not considered faces of humans. This dataset is labeled as 0.
   
2. **Data Preprocessing**: The pixel values of the images are normalized to a range between 0 and 1 to improve training efficiency.

3. **Train-Test Split**: The dataset is divided into training, validation, and testing sets:
   - **Training Set**: 64% of the total dataset
   - **Validation Set**: 16% of the total dataset
   - **Test Set**: 20% of the total dataset

3. **Model Building**: A CNN model is created using the Keras API. The architecture consists of:
   - Three convolutional layers to extract features with ReLU activation functions.
   - Max pooling layers to reduce spatial dimensions.
   - A flattening layer to convert the multi-dimensional output from the convolutional and pooling layers into a one-dimensional array. 
   - A Dense layer and an output layer with a sigmoid activation function for binary classification. Softmax activation funtion canbe used if we need mulitple classification. 

4. **Hyperparameter Tuning**: Keras Tuner is used to optimize the model's hyperparameters.
    -Number of filters in convolutional layers
    -Dense units
    -Learning rate.

6. **Training and Evaluation**: The model is trained for 10 epochs with 32 batch size. And validation done for validation dataset. Then the model is evaluated on the test set to obtain the test accuracy. A confusion matrix is generated to analyze the model's performance further:

7. **Prediction**: The trained model is used to make predictions on new images, identifying whether a face is present.

## Features
- Custom CNN architecture for efficient face detection.
- Hyperparameter tuning for optimal model performance.
- Real-time prediction on new images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sahasri/custom-face-detection.git
   cd custom-face-detection
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download dataset: The required datasets are included in the repository as zip files. 
- **faces-dataset.zip**: This zip file contains 200 images of faces of human.
- **non-faces_dataset.zip**: This zip file contains 200 images of non-faces, such as flowers, airplanes, motorcycles, nature, dogs, and cats.
- **try-test.zip**: This zip file contains mix of faces and non faces to try prediction.
- And also datasets are available in kaggle.
  ```bash
    kaggle datasets download -d sahasrimanimendra/non-faces-dataset
    kaggle datasets download -d sahasrimanimendra/faces-dataset
    kaggle datasets download -d sahasrimanimendra/try-test
## File Structure
```bash
Custom-Face-Detection-with-CNNs/                     
├── faces_dataset/                          
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...

├── non_faces_dataset/                      
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...

├── Custom_Face_Detection_with_CNNs.ipynb  
├── requirements.txt                                                    
└── README.md                    

      
