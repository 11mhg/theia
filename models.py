import tensorflow as tf
import numpy as np
from layers.resnextblock import resnextBuilder

class Resnext_Backbone(tf.keras.Model):
    def __init__(self, block_def, cardinality):
        super(Resnext_Backbone, self).__init__()

        self.network_stride = 32

        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=[7,7],
            strides=2,
            padding='SAME'
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3,3),
            strides=2,
            padding='SAME'
        )

        self.b1_layer = resnextBuilder(
            filters = 128,
            strides=1,
            groups = cardinality,
            num_blocks = block_def[0]
        )
        self.b2_layer = resnextBuilder(
            filters = 256,
            strides=2,
            groups = cardinality,
            num_blocks = block_def[1]
        )
        self.b3_layer = resnextBuilder(
            filters = 512,
            strides=2,
            groups = cardinality,
            num_blocks = block_def[2]
        )
        self.b4_layer = resnextBuilder(
            filters = 1024,
            strides=2,
            groups = cardinality,
            num_blocks = block_def[3]
        )
    
    def call(self, inputs, training=None, return_intermediate_resolutions = False):
        x = tf.nn.relu( self.bn1( self.conv1( inputs ), training=training ) )
        x = self.pool1(x)

        b1 = self.b1_layer( x , training=training )
        b2 = self.b2_layer( b1, training=training )
        b3 = self.b3_layer( b2, training=training )
        b4 = self.b4_layer( b3, training=training )
        
        if return_intermediate_resolutions:
            return b4, {
                'b1': b1,
                'b2': b2,
                'b3': b3,
                'b4': b4
            }
        return b4

def build_resnext50():
    return Resnext_Backbone([3,4,6,3], cardinality=32)

def build_resnext101():
    return Resnext_Backbone([3,4,23,3], cardinality=32)

def build_resnext152():
    return Resnext_Backbone([3,8,36,3], cardinality=32)



def get_classifier(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model