import tensorflow as tf
import numpy as np
import os
import pandas as pd

TRAIN = 'data/train_v2/'
TEST = 'data/test_v2/'
SEGMENTATION = 'data/train_ship_segmentations_v2.csv'

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, input_image, input_mask):
        input_image = self.augment_inputs(input_image)
        input_mask = self.augment_labels(input_mask)
        return input_image, input_mask

def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle is None or mask_rle == '' or pd.isnull(mask_rle):
        return np.zeros(shape + (1,))

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    img = img.reshape(shape[::-1]).T  # Flip shape dimensions before reshape and transpose
    return img[..., np.newaxis]

def load_image(image_id):
    file_path = os.path.join(TRAIN, image_id)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [768, 768])
    return img.numpy()

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image, input_mask

def get_image(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def create_generator(df):
    def generator():
        for _, group in df.groupby('ImageId'):
            image_id = group['ImageId'].values[0]
            mask_rles = group['EncodedPixels'].values

            img = load_image(image_id)

            masks = np.zeros((768, 768, 1))  # Ініціалізуємо порожню маску

            for mask_rle in mask_rles:
                if pd.notna(mask_rle):
                    mask = rle_decode(mask_rle, shape=(768, 768))
                    masks = np.maximum(masks, mask)  # Об'єднуємо маски

            img_tiles = tf.image.extract_patches(images=tf.expand_dims(img, axis=0),
                                                 sizes=[1, 256, 256, 1],
                                                 strides=[1, 256, 256, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')
            mask_tiles = tf.image.extract_patches(images=tf.expand_dims(masks, axis=0),
                                                  sizes=[1, 256, 256, 1],
                                                  strides=[1, 256, 256, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID')

            for i in range(img_tiles.shape[1]):
                for j in range(img_tiles.shape[2]):
                    if tf.reduce_sum(mask_tiles[0, i, j]) > 0:
                        yield {'file_name': image_id,
                               'image':  tf.reshape(img_tiles[0, i, j], [256, 256, 3]),
                               'segmentation_mask': tf.reshape(mask_tiles[0, i, j], [256, 256, 1])}

    return generator

def create_dataset(df):
    generator = create_generator(df)
    return tf.data.Dataset.from_generator(generator,
        output_signature={
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string),
            'image': tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8),
            'segmentation_mask': tf.TensorSpec(shape=(256, 256,1), dtype=tf.uint8),
        }
    )

def create_train_batches(df, batch_size=1, buffer_size=1):
    train_ds = create_dataset(df)
    train_images = train_ds.map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_batches = (
        train_images
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
            .repeat()
            # .map(Augment())
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    return train_batches

def create_val_batches(df, batch_size=1):
    val_ds = create_dataset(df)
    val_images = val_ds.map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_batches = val_images.batch(batch_size)
    
    return val_batches
