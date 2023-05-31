import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from metrics import weight_binary_crossentropy, f1_score, iou_score
import numpy as np

class UNetModel:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self.build_model()

    def conv_block(self, inputs, filters, kernel_size):
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def upconv_block(self, inputs, skip_features, filters, kernel_size):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.concatenate([x, skip_features], axis=3)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def attention_gate(self, x, g):
        theta_x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=False)(x)
        phi_g = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=False)(g)
        f = tf.keras.layers.add([theta_x, phi_g])
        f = tf.keras.layers.Activation('sigmoid')(f)
        x = tf.keras.layers.multiply([x, f])
        return x

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Encoding path
        conv1 = self.conv_block(inputs, 64, 3)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, 128, 3)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, 256, 3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, 512, 3)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Decoding path
        conv5 = self.conv_block(pool4, 1024, 3)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)

        up6 = self.upconv_block(drop5, drop4, 512, 2)
        up6 = self.attention_gate(up6, conv4)

        up7 = self.upconv_block(up6, conv3, 256, 2)
        up7 = self.attention_gate(up7, conv3)

        up8 = self.upconv_block(up7, conv2, 128, 2)
        up8 = self.attention_gate(up8, conv2)

        up9 = self.upconv_block(up8, conv1, 64, 2)
        up9 = self.attention_gate(up9, conv1)

        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(up9)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def summary(self):
        print(self.model.summary())

    def train(self, train_batches, val_batches, epochs=10, steps_per_epoch = 5000, batch_size=1):
        checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5', verbose=1, monitor='val_f1_score', save_best_only=True,
                                     mode='auto')

        self.model.compile(optimizer='adam',
                           loss=weight_binary_crossentropy,
                           metrics=['accuracy', iou_score, tf.keras.metrics.Recall(), f1_score])

        self.model.fit(train_batches,
                       epochs=epochs,
                       batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=val_batches,
                       callbacks=[checkpoint])

    def predict(self, image):
        # Assuming that the input tensor is of shape (786, 786, 3)
        masks = []

        # Split the image into 9 equal parts
        for i in range(3):
            for j in range(3):
                sub_image = image[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :]
                sub_image = tf.expand_dims(sub_image, axis=0)  # add batch dimension
                mask = self.model.predict(sub_image)  # assuming that model.predict returns a mask of shape (256, 256, 1)
                masks.append(mask)

        # Concatenate the masks to get a full 786x786 mask
        masks = np.array(masks)
        masks = np.reshape(masks, (3, 3, 256, 256, 1))
        masks = np.concatenate(np.concatenate(masks, axis=1), axis=1)
        return masks


#model = UNetModel()
#model.summary()