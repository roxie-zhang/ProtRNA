import tensorflow as tf
from tensorflow.keras import layers, initializers

from multihead_attention import MultiheadAttention #noqa


def gelu(x):
    """Implementation of the gelu activation function.
    """
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


class TransformerLayer(layers.Layer):
    """Transformer layer block."""

    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv=True,
            use_rotary_embeddings: bool = False,
            name=None,
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv)
    
    def _init_submodules(self, add_bias_kv):

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5)
        
        self.fc1 = layers.Dense(units=self.ffn_embed_dim, input_dim=self.embed_dim)
        self.fc2 = layers.Dense(units=self.embed_dim, input_dim=self.ffn_embed_dim)

        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5)

    def call(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
    

class RobertaLMHead(layers.Layer):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, decoder):
        super().__init__(name='lm_head')
        self.decoder = decoder
        self.dense = layers.Dense(
            units=embed_dim, 
            input_dim=embed_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=1.0),
            name="dense"
            )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.bias = self.add_weight("bias", shape=(output_dim,), initializer="zeros", trainable=True)

    def call(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = tf.matmul(x, self.decoder, transpose_b=True) + self.bias
        return x
