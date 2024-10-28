import tensorflow as tf # type: ignore

def IOU_calc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1))
    return intersection / union