import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    """Convolutional block for embedding network

    Builds single convolutional block with specified parameters.
    Includes batch normalization, makes dropout optional, and 
    uses ReLU activation function.

    Attributes
    ----------
    conv : tf.keras.layers.Conv2D
        Convolutional layer.
    batch_norm : tf.keras.layers.BatchNormalization
        Batch normalization layer.
    dropout : tf.keras.layers.Dropout
        Dropout layer, defaulting to 0% dropout.
    relu : tf.keras.layers.ReLU
        ReLU activation layer.

    Parameters
    ----------
    filters
        Number of filters for block.
    kernel_size
        Size of the filters.
    strides
        Strides to use during convolution.
    padding
        Type of padding to use.
    dropout
        Optional value denoting dropout percentage at 
        end of convolutional block.
    """
    def __init__(
        self, 
        filters: int, 
        kernel_size: int | tuple,
        strides: int = 2,
        padding: str = 'same',
        dropout: float = 0.0
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Passes input tensor through convolutional block"""
        x = self.conv(input_tensor)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block for embedding network

    Builds residual block with three convolutional layers and three
    layers of batch normalization. Uses ReLU activation function and
    sets padding to 'same' for second convolutional layer.

    Attributes
    ----------
    conv1 : tf.keras.layers.Conv2D
        First convolutional layer in residual block.
    conv2 : tf.keras.layers.Conv2D
        Second convolutional layer in residual block.
    conv3 : tf.keras.layers.Conv2D
        Third and final convolutional layer in residual block.
    bn1 : tf.keras.layers.BatchNormalization
        First batch normalization layer in residual block.
    bn2 : tf.keras.layers.BatchNormalization
        Second batch normalization layer in residual block.
    bn3 : tf.keras.layers.BatchNormalization
        Third and final batch normalization layer in residual block.
    relu : tf.keras.layers.ReLU
        ReLU activation layer.

    Parameters
    ----------
    filters
        List of number of filters for the three convolutional layers.
    kernel_size
        Size of the filters in the second convolutional layer.
    """
    def __init__(
        self,
        input_dim: int,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding='same'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding='same'
        )
        self.skip_connection = tf.keras.layers.Identity()
        if strides != 1 or input_dim != filters:
            self.skip_connection = tf.keras.layers.Conv2D(
                filters, kernel_size=1, strides=strides, padding='same'
            )

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_tensor, training=False):
        """Passes input tensor through residual block"""
        identity_x = self.skip_connection(input_tensor)
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x += identity_x
        x = self.relu(x)
        return x


class EmbeddingNetwork(tf.keras.layers.Layer):
    """Embedding network component of Siamese network

    Builds single convolutional block with specified parameters.
    Includes batch normalization, makes dropout optional, and 
    uses ReLU activation function.

    Attributes
    ----------
    blocks : list[ConvBlock, ResidualBlock]
        List of either vanilla or residual convolutional blocks.
    bn : tf.keras.layers.BatchNormalization
        Batch normalization layer.
    flatten : tf.keras.layers.Flatten
        Flattening layer.
    output_layer : tf.keras.layers.Dense
        Dense output layer of embedding network.

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
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_type: str,
        n_blocks: int,
        filters: int,
        kernel_size: int,
        strides: int = 2,
        padding: str = 'same',
        dropout: float = 0.0
    ):
        super().__init__()
        self.blocks = []
        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='relu')

        if layer_type == 'vanilla':
            for _ in range(n_blocks):
                block = ConvBlock(
                    filters=filters, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    padding=padding, 
                    dropout=dropout
                )
                self.blocks.append(block)

        elif layer_type == 'residual':
            for _ in range(n_blocks):
                block = ResidualBlock(
                    input_dim=input_dim,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides
                )
                self.blocks.append(block)

        else:
            raise ValueError("Argument `layer_type` must be one of {vanilla, residual}")
        

    def call(self, input_tensor, training=False):
        """Passes input tensor through series of convolutional blocks"""
        x = self.bn(input_tensor, training=training)
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

    