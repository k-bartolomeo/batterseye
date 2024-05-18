import tensorflow as tf
from .generator import PairGenerator


def make_dataset(
    pairs: list, 
    labels: list,
    output_types: tuple,
    output_shapes: tuple,
    batch_size: int,
    epochs: int,
    training: bool = False
) -> tf.data.Dataset:
    """Builds TensorFlow dataset from given data

    Creates instance of PairGenerator using provided image pairs and 
    labels. Generator needed because size of dataset too big for inputs
    and outputs to be saved in memory as a pair of NumPy arrays. Arrays
    for individual images can be loaded into memory as NumPy arrays and 
    stored within a list comfortably.

    Parameters
    ----------
    pairs
        List of image pairs stored as NumPy arrays.
    labels
        List of boolean values denoting whether images in pairs are 
        of same or different players. 0 indicates same player, and 1
        indicates different players.
    output_types
        Two-item tuple denoting dtypes of values returned by dataset.
        First item of tuple is dictionary with keys 'x1' and 'x2' and
        values that are TensorFlow dtypes. Second item of tuple is 
        TensorFlow dtype.
    output_shapes
        Two-item tuple denoting shapes of values returned by dataset.
        First item of tuple is dictionary with keys 'x1' and 'x2' and 
        values for (height, width, channels) of arrays returned. Second
        item of tuple is either `None` or a TensorFlow dtype.
    batch_size
        Batch size for the dataset.
    epochs
        Number of epochs for which model is trained. Used to ensure 
        that dataset is repeated sufficiently such that TensorFlow
        does not return an error suggesting the generator has run
        out of data during training.
    training
        Boolean denoting whether or not the dataset is a training or 
        test dataset. Value passed to PairGenerator initialization.

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset for given image pairs and labels. Batched
        and repeated using given parameters.
    """
    gen = PairGenerator(pairs=pairs, labels=labels, training=training)
    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_types, output_shapes=output_shapes
    )
    dataset = dataset.batch(batch_size).repeat(epochs)
    return dataset