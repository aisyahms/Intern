import tensorflow as tf


class TripletLoss:

#     https://www.tensorflow.org/addons/tutorials/losses_triplet    
#     def conv_net(self, x, reuse=False):
#         model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
#         tf.keras.layers.MaxPooling2D(pool_size=2),
#         tf.keras.layers.Dropout(0.3),
            
#         tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
#         tf.keras.layers.MaxPooling2D(pool_size=2),
#         tf.keras.layers.Dropout(0.3),
            
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
#         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
#         ])

#         return model
                                                    
                                     
# tf.contrib not available in TF 2.0, need to edit using compat.v1.layers.conv2d()   
    def conv_net(self, x, reuse=False):
        with tf.name_scope("model"): 
            with tf.compat.v1.variable_scope("conv1") as scope:
                net = tf.compat.v1.layers.conv2d(x, filters=32, kernel_size=[7, 7], activation=tf.nn.relu, padding='same',
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), reuse=reuse) 
                # reuse: whether to reuse the weights of a previous layer by the same name
                net = tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(net)

            with tf.compat.v1.variable_scope("conv2") as scope:
                net = tf.compat.v1.layers.conv2d(net, filters=64, kernel_size=[5, 5], activation=tf.nn.relu, padding='same',
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), reuse=reuse)
                net = tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(net)

            with tf.compat.v1.variable_scope("conv3") as scope:
                net = tf.compat.v1.layers.conv2d(net, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='same',
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), reuse=reuse)
                net = tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(net)

            with tf.compat.v1.variable_scope("conv4") as scope:
                net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=[1, 1], activation=tf.nn.relu, padding='same',
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), reuse=reuse)
                net = tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(net)

            with tf.compat.v1.variable_scope("conv5") as scope:
                net = tf.compat.v1.layers.conv2d(net, filters=28, kernel_size=[1, 1], activation=None, padding='same',
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), reuse=reuse)
                net = tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(net)

            net = tf.compat.v1.layers.flatten(net)

        return net


    def triplet_loss(self, model_anchor, model_positive, model_negative, margin):
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
        return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))
    