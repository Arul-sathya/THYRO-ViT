import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate the Dice Coefficient."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return (2 * intersection + smooth) / (denominator + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    """Calculate the Intersection over Union (IoU)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred)
    return (intersection + smooth) / (union + smooth)
