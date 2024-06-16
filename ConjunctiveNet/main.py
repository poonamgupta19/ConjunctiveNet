import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, LeakyReLU
from keras.optimizers import Adam
from skimage import filters
from skimage import exposure


# Load Dataset
def loadDataset(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            label = filename.split("_")[0]
            labels.append(label)
    return images, labels


train_images, train_labels = loadDataset("path_to_train_folder")
test_images, test_labels = loadDataset("path_to_test_folder")


# Pre-processing
def preprocessDataset(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter
    gaussian_filtered_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gaussian_filtered_img)

    return clahe_img


preprocessed_train_images = [preprocessDataset(img) for img in train_images]
preprocessed_test_images = [preprocessDataset(img) for img in test_images]

# Image Augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=20,
    zoom_range=0.3,
    contrast_stretching=True,
)


# Segmentation using Modified Otsu's Method
def otsuThresholding(image):
    val = filters.threshold_otsu(image)
    segmented_img = image > val
    return segmented_img


segmented_train_images = [otsuThresholding(img) for img in preprocessed_train_images]
segmented_test_images = [otsuThresholding(img) for img in preprocessed_test_images]


# Feature Extraction and Severity Detection using VGG16
def modelBuilder():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="linear")(x)
    x = LeakyReLU(alpha=0.1)(x)
    predictions = Dense(4, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


model = modelBuilder()


# Prepare data for training
def prepareData(images, labels):
    X = np.array([cv2.resize(img, (224, 224)) for img in images])
    y = np.array(labels)
    return X, y


X_train, y_train = prepareData(segmented_train_images, train_labels)
X_test, y_test = prepareData(segmented_test_images, test_labels)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
