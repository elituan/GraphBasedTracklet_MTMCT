
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"

MODEL_INPUT_SIZE = [None, 384, 384, 3]

embedding_model = tf.keras.models.Sequential([
  hub.KerasLayer(MODEL_URL, trainable=False) # EfficientNet V2 S backbone, frozen weights
])

embedding_model.build(MODEL_INPUT_SIZE)

#DATASET_NAME = 'cats_vs_dogs'
# DATASET_NAME = 'cifar10'
DATASET_NAME = 'cars196'

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


# Naming schema: <dataset_name>-<dataset_split>.<model-name>.embeddings.pickle
DST_FNAME = f'/content/drive/MyDrive/{DATASET_NAME}-train.efficientnet_v2_imagenet1k_s.embeddings.pickle'

if Path(DST_FNAME).exists():
  # When you need to use the embeddings, upload the file (or store it on Drive and mount your drive folder in Colab), then run:
  df = pd.read_pickle(DST_FNAME) # adapt the path as needed
  embeddings = np.array(df.embedding.values.tolist())
  labels = df.label.values
else:
  embeddings = []
  labels = []
  for features_batch in tqdm(batched_ds):
    embeddings.append(embedding_model(features_batch['image']).numpy())
    labels.append(features_batch['label'].numpy())

  embeddings = np.concatenate(embeddings)
  labels = np.concatenate(labels)

  # Store the precompued values to disk
  df = pd.DataFrame({'embedding':embeddings.tolist(),'label':labels})
  df.to_pickle(DST_FNAME)
  # Download the generated file to store the calculated embeddings.

NUM_CLASSES = np.unique(labels).shape[0]

# zip together embeddings and their labels, cache in memory (maybe not necessay or maybe faster this way), shuffle, repeat forever.
embeddings_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(embeddings),
    tf.data.Dataset.from_tensor_slices(labels)
)).cache().shuffle(1000).repeat()

@tf.function
def make_label_for_pair(embeddings, labels):
  #embedding_1, label_1 = tuple_1
  #embedding_2, label_2 = tuple_2
  return (embeddings[0,:], embeddings[1,:]), tf.cast(labels[0] == labels[1], tf.float32)

# because of shuffling, we can take two adjacent tuples as a randomly matched pair
train_ds = embeddings_ds.window(2, drop_remainder=True)
train_ds = train_ds.flat_map(lambda w1, w2: tf.data.Dataset.zip((w1.batch(2), w2.batch(2)))) # see https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
# generate the target label depending on whether the labels match or not
train_ds = train_ds.map(make_label_for_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# resample to the desired distribution
#train_ds = train_ds.rejection_resample(lambda embs, target: tf.cast(target, tf.int32), [0.5, 0.5], initial_dist=[0.9, 0.1])
#train_ds = train_ds.map(lambda _, vals: vals) # discard the prepended "selected" class from the rejction resample, since we aleady have it available

## Model hyperparters
EMBEDDING_VECTOR_DIMENSION = 1280
# EMBEDDING_VECTOR_DIMENSION = int(1280/2)
IMAGE_VECTOR_DIMENSIONS = 128
ACTIVATION_FN = 'tanh' # same as in paper
MARGIN = 0.005

# DST_MODEL_FNAME = f'/content/drive/MyDrive/trained_model.margin-{MARGIN}.{Path(Path(DST_FNAME).stem).stem}'
DST_MODEL_FNAME = f'trained_model.margin-{MARGIN}.{Path(Path(DST_FNAME).stem).stem}'


## These functions are straight from the Keras tutorial linked above

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def cosine_distance(vects):
    """Find the Cosine distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # NOTE: Cosine_distance = 1 - cosine_similarity
    # Cosine distance is defined betwen [0,2] where 0 is vectors with the same direction and verse,
    # 1 is perpendicular vectors and 2 is opposite vectors
    cosine_similarity = tf.keras.layers.Dot(axes=1, normalize=True)(vects)
    return 1 - cosine_similarity


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels (1 for same-class pair, 0 for different-class), fp32.
            y_pred: List of predicted distances, fp32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_dist = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_dist + (y_true) * margin_square
        )

    return contrastive_loss

from tensorflow.keras import layers, Model

emb_input_1 = layers.Input(EMBEDDING_VECTOR_DIMENSION)
emb_input_2 = layers.Input(EMBEDDING_VECTOR_DIMENSION)

# projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
projection_model = tf.keras.models.Sequential([
  layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation=ACTIVATION_FN, input_shape=(EMBEDDING_VECTOR_DIMENSION,))
])

v1 = projection_model(emb_input_1)
v2 = projection_model(emb_input_2)

computed_distance = layers.Lambda(cosine_distance)([v1, v2])
# siamese is the model we train
siamese = Model(inputs=[emb_input_1, emb_input_2], outputs=computed_distance)

## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
## We have 128 pairs for each epoch, thus in total we will have 128 x 2 x 1000 images to give to the siamese
TRAIN_BATCH_SIZE = 10000
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 50

# TODO: If there's a need to adapt the learning rate, explicitly create the optimizer instance here and pass it into compile
siamese.compile(loss=loss(margin=MARGIN), optimizer="RMSprop")
siamese.summary()

%load_ext tensorboard
%tensorboard --logdir=logs

!rm -rf logs
callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=5)
]

# TODO: Would be good to have a validation dataset too.

ds = train_ds.batch(TRAIN_BATCH_SIZE)#.prefetch(tf.data.AUTOTUNE)
history = siamese.fit(
    ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    class_weight={0:1/NUM_CLASSES, 1:(NUM_CLASSES-1)/NUM_CLASSES}
)

# Build full inference model (from image to image vector):

im_input = embedding_model.input
embedding = embedding_model(im_input)
image_vector = projection_model(embedding)
inference_model = Model(inputs=im_input, outputs=image_vector)

inference_model.save(DST_MODEL_FNAME, save_format='tf', include_optimizer=False)


def write_embeddings_for_tensorboard(image_vectors: list, labels: list, root_dir: Path):
    import csv
    from tensorboard.plugins import projector
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / 'values.tsv').open('w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(image_vectors)

    with (root_dir / 'metadata.tsv').open('w') as fp:
        for lbl in labels:
            fp.write(f'{lbl}\n')

    image_vectors = np.asarray(image_vectors)
    embeddings = tf.Variable(image_vectors, name='embeddings')
    CHECKPOINT_FILE = str(root_dir / 'model.ckpt')
    ckpt = tf.train.Checkpoint(embeddings=embeddings)
    ckpt.save(CHECKPOINT_FILE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    embedding.tensor_path = 'values.tsv'
    projector.visualize_embeddings(root_dir, config)

inference_model = tf.keras.models.load_model(DST_MODEL_FNAME, compile=False)

# NUM_SAMPLES_TO_DISPLAY = 10000
NUM_SAMPLES_TO_DISPLAY = 3000
LOG_DIR=Path('logs')
!rm -rf logs
LOG_DIR.mkdir(exist_ok=True, parents=True)

val_ds = (tfds.load(DATASET_NAME, split='test')
          .shuffle(500, seed=42)
          .take(NUM_SAMPLES_TO_DISPLAY)
          .map(resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

# compute embeddings of the images and their labels, store them in a tsv file for visualization
image_vectors = []
labels = []
for feats_batch in tqdm(val_ds):
  ims = feats_batch['image']
  lbls = feats_batch['label'].numpy()
  embs = inference_model(ims).numpy()
  image_vectors.extend(embs.tolist())
  labels.extend(lbls.tolist())

write_embeddings_for_tensorboard(image_vectors, labels, LOG_DIR)

# Do the same with some of the training data, just to see if it works with that
ds = embeddings_ds.take(NUM_SAMPLES_TO_DISPLAY).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
_image_vectors = []
_labels = []
for feats_batch in tqdm(ds):
  ims, lbls = feats_batch
  ims = ims.numpy()
  lbls = lbls.numpy()
  embs = projection_model(ims).numpy()
  _image_vectors.extend(embs.tolist())
  _labels.extend(lbls.tolist())
write_embeddings_for_tensorboard(_image_vectors, _labels, LOG_DIR/'train')

