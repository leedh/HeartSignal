import keras
import keras.layers as layers
import tensorflow as tf

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(2, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    #for filters in [64, 128, 256]:
    for filters in [4, 8, 16]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    #for filters in [256, 128, 64, 32]:
    for filters in [16, 8, 4, 2]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# class UNet(tf.keras.Model):
#     def __init__(self, input_shape=(256, 256, 3)):
#         super(UNet, self).__init__()
#         self.inputs = layers.Input(shape=input_shape)

#         # Contracting path
#         self.conv1 = self.conv_block(64, 3)
#         self.conv2 = self.conv_block(128, 3)
#         self.conv3 = self.conv_block(256, 3)
#         self.conv4 = self.conv_block(512, 3)
#         self.conv5 = self.conv_block(1024, 3, max_pool=False)  # No pooling after this block

#         # Expanding path
#         self.upconv6 = self.upconv_block(512, 2)
#         self.conv6 = self.conv_block(512, 3, max_pool=False)
        
#         self.upconv7 = self.upconv_block(256, 2)
#         self.conv7 = self.conv_block(256, 3, max_pool=False)
        
#         self.upconv8 = self.upconv_block(128, 2)
#         self.conv8 = self.conv_block(128, 3, max_pool=False)
        
#         self.upconv9 = self.upconv_block(64, 2)
#         self.conv9 = self.conv_block(64, 3, max_pool=False)

#         self.outputs = layers.Conv2D(2, kernel_size=1, activation='sigmoid')
        
#     # 모델의 입력과 출력을 설정
#         self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name="u-net")

#     def call(self, inputs):
#         return self.model(inputs)

#     def conv_block(self, filters, kernel_size, max_pool=True):
#         """Constructs a block with two convolutions, optional max pooling."""
#         def f(x):
#             x = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
#             x = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
#             if max_pool:
#                 x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#             return x
#         return f

#     def upconv_block(self, filters, kernel_size):
#         """Constructs a block for Transpose Convolution."""
#         def f(x):
#             x = layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), activation='relu', padding='same')(x)
#             return x
#         return f

#     def call(self, inputs):
#         # Contracting path
#         c1 = self.conv1(inputs)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         c5 = self.conv5(c4)

#         # Expanding path
#         u6 = self.upconv6(c5)
#         u6 = layers.concatenate([u6, c4], axis=-1)
#         c6 = self.conv6(u6)
        
#         u7 = self.upconv7(c6)
#         u7 = layers.concatenate([u7, c3], axis=-1)
#         c7 = self.conv7(u7)
        
#         u8 = self.upconv8(c7)
#         u8 = layers.concatenate([u8, c2], axis=-1)
#         c8 = self.conv8(u8)
        
#         u9 = self.upconv9(c8)
#         u9 = layers.concatenate([u9, c1], axis=-1)
#         c9 = self.conv9(u9)

#         outputs = self.outputs(c9)
#         return outputs

# class u_net(tf.keras.Model):
#     def __init__(self, input_shape):
#         super(u_net, self).__init__()  # 부모 클래스의 생성자 호출
#         inputs = layers.Input(shape=input_shape)

#         # Encoder
#         x_0 = inputs
#         x_1, x_1_maxpool = self.conv_block(x_0, 64, kernel_size=3, pool_strides=(2,2))
#         x_2, x_2_maxpool = self.conv_block(x_1_maxpool, 128, kernel_size=3, pool_strides=(2,2))
#         x_3, x_3_maxpool = self.conv_block(x_2_maxpool, 256, kernel_size=3, pool_strides=(2,2))
#         x_4, x_4_maxpool = self.conv_block(x_3_maxpool, 512, kernel_size=3, pool_strides=(2,2))
#         x_5, _ = self.conv_block(x_4_maxpool, 1024, kernel_size=3, maxpooling=False)

#         # Decoder
#         x_6 = self.residual_block(x_5, x_4, 512, 2, cropping=(4,4))
#         x_7, _ = self.conv_block(x_6, 512, kernel_size=3, maxpooling=False)
#         x_8 = self.residual_block(x_7, x_3, 256, 2, cropping=(16,16))
#         x_9, _ = self.conv_block(x_8, 256, kernel_size=3, maxpooling=False)
#         x_10 = self.residual_block(x_9, x_2, 128, 2, cropping=(40,40))
#         x_11, _ = self.conv_block(x_10, 128, kernel_size=3, maxpooling=False)
#         x_12 = self.residual_block(x_11, x_1, 64, 2, cropping=(88,88))
#         x_13, _ = self.conv_block(x_12, 64, kernel_size=3, maxpooling=False)

#         # Output
#         outputs = layers.Conv2D(2, kernel_size=1)(x_13)
        
#         # 모델의 입력과 출력을 설정
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net")

#     def call(self, inputs):
#         return self.model(inputs)

#     def conv_block(self, input_tensor, num_filters, kernel_size, maxpooling=True, pool_size=(2, 2), pool_strides=(2, 2)):
#         """
#         A convolutional block consisting of 2 convolutional layers each followed by batch normalization and ReLU activation.
#         Optionally, a max pooling layer can be applied at the end.

#         Parameters:
#         - input_tensor: The input tensor.
#         - num_filters: Number of filters for the convolutional layers.
#         - kernel_size: Size of the convolution kernel.
#         - strides: Strides for the convolution operations.
#         - maxpooling: Whether to apply max pooling after the convolutional layers.
#         - pool_size: The size of the max pooling window.
#         - pool_strides: The stride of the max pooling operation.

#         Returns:
#         - A tuple of (last_conv_layer, pooled_layer (optional))
#         """
#         x = layers.Conv2D(num_filters, kernel_size, padding='same')(input_tensor)
#         # x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
#         # x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         if maxpooling:
#             pooled = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')(x)
#             return x, pooled
#         else:
#             return x, None

#     def residual_block(self, input_tensor, skip_tensor, num_filters, strides, cropping):
#         """
#         A residual block that adds a skip connection around the convolutional layers.

#         Parameters:
#         - input_tensor: The input tensor for the convolutional operations.
#         - skip_tensor: The tensor to be added (skipped connection) to the output of the convolutional block.
#         - num_filters: Number of filters for the convolutional layers.
#         - strides: Strides for the convolution operations.
#         - cropping: Tuple specifying how many units should be cropped from the skip_tensor before addition.

#         Returns:
#         - The output tensor of the residual block.
#         """
#         x = layers.Conv2DTranspose(num_filters, kernel_size=2, strides=strides, padding='same')(input_tensor)
#         # x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         # Adjusting the skip_tensor dimensions to match the input_tensor dimensions
#         cropped_skip_tensor = layers.Cropping2D(cropping=cropping)(skip_tensor)
#         x = layers.concatenate([x, cropped_skip_tensor], axis=-1)
        
#         # cropped_skip_tensor = layers.Cropping2D(cropping)(skip_tensor)
#         # x = layers.concatenate([x, cropped_skip_tensor], axis=-1)

#         return x    

# class u_net(tf.keras.Model):
#     def __init__(self, input_shape):
#         inputs = layers.Input(shape=input_shape)

#         # Encoder
#         x_0 = inputs
#         x_1, x_1_maxpool = self.conv_block(x_0, 64, kernel_size=3, strides=(2,2)) # [1]
#         x_2, x_2_maxpool = self.conv_block(x_1_maxpool, 128, kernel_size=3, strides=(2,2)) # [2]
#         x_3, x_3_maxpool = self.conv_block(x_2_maxpool, 256, kernel_size=3, strides=(2,2)) # [3]
#         x_4, x_4_maxpool = self.conv_block(x_3_maxpool, 512, kernel_size=3, strides=(2,2)) # [4]
#         x_5, _ = self.conv_block(x_4_maxpool, 1024, kernel_size=3, strides=(2,2), maxpooling=False) # [5]

#         # Skip connection & Decoder
#         x_6 = self.residual_block(x_5, x_4, 512, 2, cropping=(4,4)) # [6]
#         x_7, _ = self.conv_block(x_6, 512, kernel_size=3, maxpooling=False) # [7]
#         x_8 = self.residual_block(x_7, x_3, 256, 2, cropping=(16,16)) # [8]
#         x_9, _ = self.conv_block(x_8, 256, kernel_size=3, maxpooling=False) # [9]
#         x_10 = self.residual_block(x_9, x_2, 128, 2, cropping=(40,40)) # [10]
#         x_11, _ = self.conv_block(x_10, 128, kernel_size=3, maxpooling=False) # [11]
#         x_12 = self.residual_block(x_11, x_1, 64, 2, cropping=(88,88)) # [12]
#         x_13, _ = self.conv_block(x_12, 64, kernel_size=3, maxpooling=False) #[13]

#         # Output
#         outputs = layers.Conv2D(2, kernel_size=1)(x_13)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net")

#     def conv_block(self, input_tensor, num_filters, kernel_size, strides=(1, 1), maxpooling=True, pool_size=(2, 2), pool_strides=(2, 2)):
#         """
#         A convolutional block consisting of 2 convolutional layers each followed by batch normalization and ReLU activation.
#         Optionally, a max pooling layer can be applied at the end.

#         Parameters:
#         - input_tensor: The input tensor.
#         - num_filters: Number of filters for the convolutional layers.
#         - kernel_size: Size of the convolution kernel.
#         - strides: Strides for the convolution operations.
#         - maxpooling: Whether to apply max pooling after the convolutional layers.
#         - pool_size: The size of the max pooling window.
#         - pool_strides: The stride of the max pooling operation.

#         Returns:
#         - A tuple of (last_conv_layer, pooled_layer (optional))
#         """
#         x = layers.Conv2D(num_filters, kernel_size, strides=strides, padding='same')(input_tensor)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         x = layers.Conv2D(num_filters, kernel_size, strides=strides, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         if maxpooling:
#             pooled = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')(x)
#             return x, pooled
#         else:
#             return x, None

#     def residual_block(self, input_tensor, skip_tensor, num_filters, strides, cropping):
#         """
#         A residual block that adds a skip connection around the convolutional layers.

#         Parameters:
#         - input_tensor: The input tensor for the convolutional operations.
#         - skip_tensor: The tensor to be added (skipped connection) to the output of the convolutional block.
#         - num_filters: Number of filters for the convolutional layers.
#         - strides: Strides for the convolution operations.
#         - cropping: Tuple specifying how many units should be cropped from the skip_tensor before addition.

#         Returns:
#         - The output tensor of the residual block.
#         """
#         x = layers.Conv2DTranspose(num_filters, kernel_size=2, strides=strides, padding='same')(input_tensor)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)

#         # Adjusting the skip_tensor dimensions to match the input_tensor dimensions
#         cropped_skip_tensor = layers.Cropping2D(cropping=cropping)(skip_tensor)
#         x = layers.concatenate([x, cropped_skip_tensor], axis=-1)
        
#         # cropped_skip_tensor = layers.Cropping2D(cropping)(skip_tensor)
#         # x = layers.concatenate([x, cropped_skip_tensor], axis=-1)

#         return x    