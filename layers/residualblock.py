import tensorflow as tf
import numpy as np


class ResidualBlock(tf.keras.Model):

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels  = input_channels
        self.output_channels = output_channels

        self.stride = stride

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(output_channels//4, [1,1], strides=1, 
            padding='SAME', use_bias=False)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(output_channels//4, [3,3], strides=self.stride, 
            padding='SAME', use_bias=False)
            
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(output_channels, [1,1], strides=1, 
            padding='SAME', use_bias=False)
        
        if self.stride != 1 or self.input_channels!=self.output_channels:
            self.conv4 = tf.keras.layers.Conv2D(output_channels, [1,1], strides=self.stride, 
                padding='SAME', use_bias=False)
    
    def call(self, x, training=None):
        residual = x
        x = self.conv1( self.relu( self.bn1(x, training=training) ) )
        x = self.conv2( self.relu( self.bn2(x, training=training) ) )
        x = self.conv3( self.relu( self.bn3(x, training=training) ) )
        if self.stride != 1 or self.input_channels!=self.output_channels:
            residual = self.conv4(residual)
        x += residual
        return x



if __name__ == "__main__":
    RB = ResidualBlock(3, 32, stride=1)
    print(RB(np.random.rand(1,448,448,3)).shape)
    RB = ResidualBlock(3, 32, stride=2)
    print(RB(np.random.rand(1,448,448,3)).shape)
    RB = ResidualBlock(32, 32, stride=3)
    print(RB(np.random.rand(1,448,448,32)).shape)