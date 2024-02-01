import tensorflow.keras.layers as layers
import tensorflow as tf

class u_net(tf.keras.Model):
    def __init__(self, input_shape):
        inputs = layers.Input(shape=input_shape)

        # Encoder
        x_0 = inputs
        x_1, x_1_maxpool = conv_block(x_0, 64, kernel_size=3, strides=(2,2)) # [1]
        x_2, x_2_maxpool = conv_block(x_1_maxpool, 128, kernel_size=3, strides=(2,2)) # [2]
        x_3, x_3_maxpool = conv_block(x_2_maxpool, 256, kernel_size=3, strides=(2,2)) # [3]
        x_4, x_4_maxpool = conv_block(x_3_maxpool, 512, kernel_size=3, strides=(2,2)) # [4]
        x_5, _ = conv_block(x_4_maxpool, 1024, kernel_size=3, strides=(2,2), maxpooling=False) # [5]

        # Skip connection & Decoder
        x_6 = residual_block(x_5, x_4, 512, 2, cropping=(4,4)) # [6]
        x_7, _ = conv_block(x_6, 512, kernel_size=3, maxpooling=False) # [7]
        x_8 = residual_block(x_7, x_3, 256, 2, cropping=(16,16)) # [8]
        x_9, _ = conv_block(x_8, 256, kernel_size=3, maxpooling=False) # [9]
        x_10 = residual_block(x_9, x_2, 128, 2, cropping=(40,40)) # [10]
        x_11, _ = conv_block(x_10, 128, kernel_size=3, maxpooling=False) # [11]
        x_12 = residual_block(x_11, x_1, 64, 2, cropping=(88,88)) # [12]
        x_13, _ = conv_block(x_12, 64, kernel_size=3, maxpooling=False) #[13]

        # Output
        outputs = layers.Conv2D(2, kernel_size=1)(x_13)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net")

