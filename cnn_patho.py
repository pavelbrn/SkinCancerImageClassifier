import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


RANDOM_STATE = 98

tumor_classes = 2

# Input image dimensions are a 224 x 224 Matrix and 3 channels(RGB colors)
first_input = (120,120,3)

# Define the image directory:
# alternaive test path: data_experiment/train/benign/     and     data_experiment/train/malignant/
image_dir_benign = 'data/train/benign/'
image_dir_malignant = 'data/train/malignant/'


def extract_images(image_directory: str):
    ''' This function returns a list of images'''
    
    image_dir=os.listdir(image_directory)
    show_list = []
    for img in image_dir:
        if img != '.DS_Store':
            ext_img = image.load_img(str(image_directory+img), target_size=(120, 120))
            show_list.append(ext_img)
    #show_img = ext_img
    return show_list

# Extract two arrays of images, benign and malignant:
benign_images = extract_images(image_dir_benign)
malignant_images = extract_images(image_dir_malignant)

# Create X_malignant
def create_feature_label_array(image_list):
    '''Return a numpy array of all images'''
    feature_images= []
    for image in image_list:
        data = np.asarray( image )
        feature_images.append(data)

    feature_images = np.array(feature_images)
    return feature_images


# Create labels where each malignant image is labeled as 1
X_malignant = create_feature_label_array(malignant_images)
y_malignant = np.ones(len(X_malignant))

# Show the shapes of the features and labels
X_malignant.shape, y_malignant.shape

# Create labels where each benign skin leasion is labeled as 0
X_benign = create_feature_label_array(benign_images)
y_benign = np.zeros(len(X_benign))
X_benign.shape,y_benign.shape

# Concatenate both malignant and benign features/labels, show their shape
X = np.concatenate((X_malignant,X_benign))
y = np.concatenate((y_malignant,y_benign))



np_shuffler = np.random.permutation(len(X))
X = X[np_shuffler]
y = y[np_shuffler]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

X_train.shape,X_test.shape, y_train.shape, y_test.shape



# define the keras model that will be used to train the data
model = keras.Sequential(
    [
        # shape is the input shape and the first layer is a convolutional layer
        keras.Input(shape=first_input),
        # create a convolution layer with 32 filters, each 3x3
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

        # create a max pooling layer with pool size 2x2
        layers.MaxPooling2D(pool_size=(2, 2)),

        # create a convolution layer with 64 filters, each 3x3
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        # create a max pooling layer with pool size 2x2
        layers.MaxPooling2D(pool_size=(2, 2)),
        

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the 2D arrays of the feature maps into 1D vectors
        layers.Flatten(),
        layers.Dropout(0.4),
        # num_classes is the number of classes we want to predict
        # the output will be a vector of length num_classes
        # the output layer will have this many nodes: num_classes
        layers.Dense(1, activation="relu"),
    ]
)

print("docker commit test")

model.summary()


batch_size = 100
epochs = 1

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


model.save("docker_alpha_model.h5")