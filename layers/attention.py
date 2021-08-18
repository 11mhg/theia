import tensorflow as tf



class FastMultiHeadedAttention(tf.keras.Model):
    """
    Creates a model which performs the transformer based attention when given the query, value, and keys. 
    Self attention is performed by using the same tensor for all three. The value and keys are also typically the same tensor shape.
    Note that this model assumes that these are from images with shape [batch, W, H, num filters].
    The num_filters and the number of heads should be known ahead of time (note that num_filters % num_heads == 0). 

    This version performs a larger dense prior to splitting the number of heads, meaning that with the same number of parameters performing a very
    similar attention is much faster as it does not require a for loop (The scaled_dot_product_attention is broadcasted for speed).

    Parameters:
    num_filters (int): The number of filters in the input tensor (typically the depth of a tensor).
    num_heads   (int): The number of attention heads to perform attention. (note that num_filters % num_heads == 0). 
    name     (string): The name of the layer.
    """
    def __init__(self, num_filters, num_heads):
        super(FastMultiHeadedAttention, self).__init__()
 
        assert num_filters % num_heads == 0, "num_filters % num_heads must equal 0"
        self.query_size = num_filters // num_heads
        self.key_size   = num_filters // num_heads
        self.value_size = num_filters // num_heads
        self.num_filters = num_filters
 
        self.h = num_heads
 
        self.wq = tf.keras.layers.Dense(num_filters) 
        self.wk = tf.keras.layers.Dense(num_filters)
        self.wv = tf.keras.layers.Dense(num_filters)
 
        self.wo = tf.keras.layers.Dense(num_filters)
    
    @tf.function
    def scaled_dot_product_attention(self,q, k, v, mask):
        QK = tf.matmul(q, k , transpose_b=True)
        scaled_attention_logits = QK/tf.math.sqrt(tf.cast( self.key_size, tf.float32))
 
        if mask is not None:
            scaled_attention_logits += (mask * -1e9) #make it super negative where the mask is positive in order to make those probs approach zero
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
 
        return output, attention_weights
 
    @tf.function
    def call(self, query, value, keys, mask=None, return_attention_scores=False):
        """
        Performs the transformer based attention when given the query, value, and keys. 
        Self attention is performed by using the same tensor for all three. The value and keys are also typically the same tensor shape.
        Note that this function assumes that these are from images with shape [batch, W, H, num filters]

        Parameters:
        query (tf.tensor): The Query for the attention.
        value (tf.tensor): The Value for the attention.
        keys  (tf.tensor): The Keys for the attention.
        mask  (tf.tensor): Optional. A mask for the attention. 
        Returns:
        (tf.tensor, tf.tensor): Returns a tuple with the output of the attention at ind 0 and the attention weights at ind 1.

        """
        query_shape = query.shape #[ batch, H, W, C]
        value_shape = value.shape #[ batch, H, W, C]
        keys_shape = keys.shape #[ batch, H, W, C]
 
        query = tf.reshape( query, [-1, query_shape[1]*query_shape[2], query_shape[-1]])
        value = tf.reshape( value, [-1, value_shape[1]*value_shape[2], value_shape[-1]])
        keys  = tf.reshape( keys , [-1, keys_shape[1] * keys_shape[2],  keys_shape[-1]])
 
        query = self.wq(query)
        value = self.wv(value)
        keys  = self.wk(keys)
        
        query = tf.reshape(query, [-1, query_shape[1]*query_shape[2], self.h, self.query_size]) #reshape to [batch, H*W, self.h, C//h]
        value = tf.reshape(value, [-1, value_shape[1]*value_shape[2], self.h, self.value_size]) #reshape to [batch, H*W, self.h, C//h]
        keys  = tf.reshape(keys , [-1, keys_shape[1] * keys_shape[2], self.h, self.key_size  ]) #reshape to [batch, H*W, self.h, C//h]
 
        query = tf.transpose(query, [0,2,1,3]) #[batch, self.h, H*W, C//h]
        value = tf.transpose(value, [0,2,1,3]) #[batch, self.h, H*W, C//h]
        keys = tf.transpose( keys , [0,2,1,3]) #[batch, self.h, H*W, C//h]
 
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, keys, value, mask) 
 
        scaled_attention = tf.transpose(scaled_attention, [0,2,1,3])#[batch, len_q, num_heads, depth ]
 
        concat_attention = tf.reshape( scaled_attention, [-1, query_shape[1]*query_shape[2], self.num_filters]) #[batch, len_q, num_filters]
 
        output = self.wo(concat_attention)
 
        output = tf.reshape(output, [-1, query_shape[1], query_shape[2], query_shape[3]]) #
        attention_weights = tf.reshape(attention_weights, [-1,self.h, query_shape[1], query_shape[2], query_shape[1]*query_shape[2]])

        if return_attention_scores:
            return output, attention_weights
        
        return output



if __name__ == "__main__":
    import numpy as np

    mha = FastMultiHeadedAttention(256, 8)
    inp = np.random.rand(4, 8, 8, 256)
    out = mha(inp, inp, inp)
    for elem in out:
        print(elem.shape)

    tf_inp = tf.keras.layers.Input(shape=(8,8,256))
    layer = FastMultiHeadedAttention(256, 8)
    out, attn_weights = layer(tf_inp, tf_inp, tf_inp, return_attention_scores = True)

    model = tf.keras.models.Model(inputs=tf_inp, outputs=out)

    print(model(inp).shape)

    model.summary()

    