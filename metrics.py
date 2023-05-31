import tensorflow as tf
from tensorflow.keras import backend as K

def weight_binary_crossentropy(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    penalty = 10.0
    logloss = tf.where(y_true == 1, penalty * y_true * tf.math.log(y_pred + K.epsilon()),
                       y_true * tf.math.log(y_pred + K.epsilon()) + (1 - y_true) * tf.math.log(
                           1 - y_pred + K.epsilon()))
    return -tf.reduce_mean(logloss)


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred >= 0.5, tf.bool)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), tf.float32))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1


def iou_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred >= 0.5, tf.bool)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(y_true, y_pred), tf.float32))
    iou = intersection / union
    return iou