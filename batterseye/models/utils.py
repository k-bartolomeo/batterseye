import tensorflow as tf


def get_distance_fn(kind: str = 'euclidean'):
    if kind == 'euclidean':
        def _(vectors):
            x, y = vectors
            squared_sum = tf.keras.ops.sum(
                tf.keras.ops.square(x - y), axis=1, keepdims=True
            )
            distance = tf.keras.ops.sqrt(
                tf.keras.ops.maximum(squared_sum, tf.keras.backend.epsilon())
            )
            return distance
        return _
    
    if kind == 'cosine':
        def _(vectors):
            x, y = vectors
            norm_x = tf.math.l2_normalize(x)
            norm_y = tf.math.l2_normalize(y)
            similarity = tf.reduce_sum(tf.multiply(norm_x, norm_y))
            return similarity
        return _
    
    raise ValueError('Argument `kind` must be one of {euclidean, cosine}')