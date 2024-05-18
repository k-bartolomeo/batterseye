import tensorflow as tf


def contrastive_loss(margin: int = 1) -> callable:
    """Builds contrastive loss function for training

    Parameters
    ----------
    margin
        Threshold under which distance between two vectors must fall
        for loss to equal 0.

    Returns
    -------
    callable
        Contrastive loss function for use in training Siamese network.
    """
    def _(y_true, y_pred):
        """Computes contrastive loss for given margin"""
        squared_pred = tf.keras.ops.square(y_pred)
        squared_margin = tf.keras.ops.square(
            tf.keras.ops.maximum(margin - (y_pred), 0)
        )
        loss = tf.keras.ops.mean(
            (1 - y_true) * squared_pred + y_true * squared_margin
        )
        return loss
    
    return _
