import tensorflow as tf
from .embedding import EmbeddingNetwork
from .utils import get_distance_fn


class SiameseNetwork(tf.keras.Model):
    """Siamese network implementation

    Builds single embedding network whose parameters are shared 
    between both pathways in Siamese network architecture. One
    pathway is for one image in given image pair, and the other 
    pathway is for the other image. Computes distance between
    different pathways' embeddings using either Euclidean distance
    or cosine similarity. Outputs probability that image pair 
    contains images of two different players.

    Attributes
    ----------
    embedding_network : EmbeddingNetowrk
        Embedding network backbone of Siamese network.
    distance : Callable
        Distance function used for training.
    merge_layer : tf.keras.layers.Lambda
        Layer computing distances between two image embeddings.
    bn : tf.keras.layers.BatchNormalization
        Batch normalization layer.
    output_layer : tf.keras.layers.Dense
        Dense output layer for binary classification.

    Parameters
    ----------
    output_dim
        Dimensionality of the final image embedding.
    layer_type
        Type of layer - vanilla or residual - used in network.
    n_blocks
        Number of convolutional blocks to use in network.
    filters
        Number of filters for each block in network.
    kernel_size
        Size of the filters in the network's blocks.
    strides
        Strides to use in vanilla convolutional blocks.
    padding
        Type of padding to use in vanilla convolutional blocks.
    dropout
        Optional value denoting dropout percentage at 
        end of vanilla convolutional block.
    distance
        Distance metric to use for training. Either 'euclidean'
        for Euclidean distance or 'cosine' for cosine similarity.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_type: str,
        n_blocks: int,
        filters: int,
        kernel_size: int | tuple,
        strides: int = 2,
        padding: str = 'same',
        dropout: float = 0.0,
        distance: str = 'euclidean'
    ):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = EmbeddingNetwork(
            input_dim=input_dim,
            output_dim=output_dim, 
            layer_type=layer_type, 
            n_blocks=n_blocks,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dropout=dropout
        )
        self.distance = get_distance_fn(kind=distance)
        self.merge_layer = tf.keras.layers.Lambda(self.distance, output_shape=(1,))
        self.bn = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input_tensor, training=False):
        """Passes pair input tensors through Siamese network"""
        input_tensor_1 = input_tensor['x1']
        input_tensor_2 = input_tensor['x2']
        x1 = self.embedding_network(input_tensor_1)
        x2 = self.embedding_network(input_tensor_2)
        x = self.merge_layer([x1, x2])
        x = self.bn(x, training=training)
        x = self.output_layer(x)
        return x

