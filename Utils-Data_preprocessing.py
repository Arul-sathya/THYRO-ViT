import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize_images(images):
    """Normalize images to the range [0, 1]."""
    return images / 255.0

def augment_images(images):
    """Apply data augmentation to images."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    return datagen.flow(images, batch_size=len(images), shuffle=False).next()

def split_dataset(images, labels, validation_split=0.2):
    """Split dataset into training and validation sets."""
    total_samples = len(images)
    val_size = int(total_samples * validation_split)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    return (images[train_indices], labels[train_indices],
            images[val_indices], labels[val_indices])
