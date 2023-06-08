!pip install -q tfds-nightly tensorboard-plugin-profile
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"

MODEL_INPUT_SIZE = [None, 384, 384, 3]

embedding_model = tf.keras.models.Sequential([
  hub.KerasLayer(MODEL_URL, trainable=False) # EfficientNet V2 S backbone, frozen weights
])

embedding_model.build(MODEL_INPUT_SIZE)

#DATASET_NAME = 'cats_vs_dogs'
DATASET_NAME = 'cifar10'
#DATASET_NAME = 'cars196'

#NOTE: For cars196 & other datasets with many classes, the rejection resampling
#      used to balance the positive and negative classes does NOT work anymore! (the input pipeline chokes)
#      Need to find a better solution!
# -> FIX: Using class weights based on the number of labels in the original dataset seems to work perfectly well (and training speed improves greatly too)

# Load dataset in a form already consumable by Tensorflow
ds = tfds.load(DATASET_NAME, split='train')

# Resize images to the model's input size and normalize to [0.0, 1.0] as per the
# expected image input signature: https://www.tensorflow.org/hub/common_signatures/images#input
def resize_and_normalize(features):
  return {
      #'id': features['id'],
      'label': features['label'],
      'image': tf.image.resize( tf.image.convert_image_dtype(features['image'], tf.float32), MODEL_INPUT_SIZE[1:3])
  }

ds = ds.map(resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

# Add batch and prefetch to dataset to speed up processing
BATCH_SIZE=256
batched_ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset has keys "id" (that we ignore), "image" and "label".
# "image" has shape [BATCH_SIZE,32,32,3] and is an RGB uint8 image
# "label" has shape [BATCH_SIZE,1] and is an integer label (value between 0 and 9)
