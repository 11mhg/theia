import tensorflow as tf
try:
    from groupconv import GroupConv2D
except Exception as e:
    from layers.groupconv import GroupConv2D


class ResnextBottleneck(tf.keras.Model):

    def __init__(self, filters, groups, strides=1):
        super(ResnextBottleneck, self).__init__()
        self.filters = filters
        self.groups  = groups
        self.strides = strides

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(self.filters, [1,1], strides=1, 
            padding='SAME', use_bias=False)
        
        self.group_conv = GroupConv2D(input_channels = filters, 
                                      output_channels = filters,
                                      kernel_size=[3,3],
                                      strides = self.strides,
                                      padding='SAME',
                                      use_bias=False,
                                      groups=self.groups)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(2*filters, [1,1], strides=1, 
            padding='SAME', use_bias=False)
            
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.residual_conv = tf.keras.layers.Conv2D(filters=2*filters,
            kernel_size=[1,1],
            strides=self.strides,
            padding="SAME")
        self.residual_bn = tf.keras.layers.BatchNormalization()
    
    def call(self, x, training=None):
        residual = x
        x = self.relu( self.bn1( self.conv1(residual), training=training ) )
        x = self.relu( self.bn2( self.group_conv(x)  , training=training ) )
        x = self.relu( self.bn3( self.conv2( x )     , training=training ) )

        residual = self.residual_bn( self.residual_conv( residual ), training=training )

        x = self.relu( x + residual )
        
        return x

def resnextBuilder(filters, strides, groups, num_blocks):
    block = tf.keras.Sequential()
    block.add(
        ResnextBottleneck(filters=filters,
                            groups=groups,
                            strides=strides)
    )

    for _ in range(1, num_blocks):
        block.add(
            ResnextBottleneck(
                filters=filters,
                strides=1,
                groups = groups
            )
        )
    
    return block



if __name__ == "__main__":
    import numpy as np

    RB = ResnextBottleneck(32, 4, strides=1)
    print(RB(np.random.rand(1,448,448,3)).shape)
    RB = ResnextBottleneck(32, 1, strides=2)
    print(RB(np.random.rand(1,448,448,3)).shape)
    RB = ResnextBottleneck(32, 1, strides=3)
    print(RB(np.random.rand(1,448,448,32)).shape)