# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:59:23 2021

@author: Tengfei Feng
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,GlobalAveragePooling1D,Dropout,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def _bn_relu(layer, dropout=0, **params):
    layer = Activation('relu')(layer)

    if dropout > 0:
        layer = Dropout(0.5)(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from tensorflow.keras.layers import Conv1D 
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same')(layer)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from tensorflow.keras.layers import Add 
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length,padding="same")(layer)
    zero_pad = (block_index % 4) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(2):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=0.5 if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            16,
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
        layer= BatchNormalization()(layer)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / 4) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params['conv_filter_length'],
        32,
        subsample_length=1,
        **params)
    layer = BatchNormalization()(layer)
    layer = _bn_relu(layer, **params)
    conv_subsample_lengths = [1,2,1,2,1,2,1,2]
    for index, subsample_length in enumerate(conv_subsample_lengths):
        num_filters = get_num_filters_at_index(
            index, 32, **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                    "embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
                    )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
    
def build_model(**params):
    num_heads = 8
    ff_dim = 32
    embed_dim = 64
    # inputs = Input(shape=(800,2), name="inputs")
    inputs = Input(shape=(600,1), name="inputs")
    x = add_resnet_layers(inputs,conv_filter_length=16)
    embedding_layer = PositionEmbedding(50, embed_dim)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    x1 = transformer_block(x,training=True)
    
    x1 = GlobalAveragePooling1D()(x1) 
    
    x1 = Dense(32, activation='relu')(x1)
    # x1 = Concatenate()([x1,inputs2_1])
    x1 = layers.Dropout(0.1)(x1)
    x1 = Dense(1)(x1)
    
    
    
    
    outputs1 = Activation('sigmoid',name='out1')(x1)
    
    
    
    model = Model(inputs=[inputs], outputs=[outputs1])
    
    return model
