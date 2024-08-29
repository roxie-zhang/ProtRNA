from typing import Dict, Optional, Tuple


import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers

from protrna.rotary_embedding import RotaryEmbedding

'''
Main divergence from the original implementation in esm repo include:
1. not using incremental_state (possibly suffering efficiency)
2. not using onnx_trace
3. device management in tensorflow differs from torch
4. not accomodating static analysis tools
'''

class MultiheadAttention(layers.Layer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            self_attention: bool = False,
            encoder_decoder_attention: bool = False,
            use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.q_proj = layers.Dense(units=embed_dim, input_dim=embed_dim, use_bias=bias)
        self.k_proj = layers.Dense(units=embed_dim, input_dim=self.kdim, use_bias=bias)
        self.v_proj = layers.Dense(units=embed_dim, input_dim=self.vdim, use_bias=bias)

        self.out_proj = layers.Dense(units=embed_dim, input_dim=embed_dim, use_bias=bias)

        if add_bias_kv:
            self.bias_k = tf.Variable(tf.zeros((1, 1, embed_dim)))
            self.bias_v = tf.Variable(tf.zeros((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def call(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(sequence length) x Batch x Channel(embed_dim)

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert query.shape == (tgt_len, bsz, embed_dim)

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(value)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = tf.concat([k, tf.repeat(self.bias_k, [1, bsz, 1])]) #shape: (src_len, bsz, embed_dim)
            v = tf.concat([v, tf.repeat(self.bias_v, [1, bsz, 1])])
            if attn_mask is not None:
                attn_mask = tf.concat(
                    [attn_mask, tf.zeros([attn_mask.shape[0], 1], dtype=attn_mask.dtype)], axis=1
                )
            if key_padding_mask is not None:
                key_padding_mask = tf.concat(
                    [
                        key_padding_mask, #shape: (bsz, src_len)
                        tf.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype),
                    ],
                    axis=1,
                )
        
        q = tf.transpose(tf.reshape(q, [tgt_len, bsz * self.num_heads, self.head_dim]), perm=[1, 0, 2])
        if k is not None:
            k = tf.transpose(tf.reshape(k, [-1, bsz * self.num_heads, self.head_dim]), perm=[1, 0, 2])
        if v is not None:
            v = tf.transpose(tf.reshape(v, [-1, bsz * self.num_heads, self.head_dim]), perm=[1, 0, 2])

        # q shape: (bsz * self.num_heads, tgt_len, self.head_dim)
        # k,v shape: (bsz * self.num_heads, src_len, self.head_dim)

        assert k is not None
        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if self.add_zero_attn: # not used
            assert v is not None
            src_len += 1
            k = tf.concat([k, tf.zeros_like(k[:, :1, :])], axis=1)
            v = tf.concat([v, tf.zeros_like(v[:, :1, :])], axis=1)
            if attn_mask is not None:
                attn_mask = tf.concat(
                    [attn_mask, tf.zeros([attn_mask.shape[0], 1])], axis=1
                )
            if key_padding_mask is not None:
                key_padding_mask = tf.concat(
                    [
                        key_padding_mask, 
                        tf.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype)
                    ],
                    axis=1
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = tf.linalg.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
        assert attn_weights.shape == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = tf.expand_dims(attn_mask, axis=0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = tf.reshape(attn_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_weights = tf.where(
                tf.cast(tf.expand_dims(tf.expand_dims(key_padding_mask, 1), 2), dtype=tf.bool), 
                float("-inf"), 
                attn_weights
            ) 
            attn_weights = tf.reshape(attn_weights, [bsz * self.num_heads, tgt_len, src_len])

        if before_softmax:
            return attn_weights, v
        
        attn_weights_float = tf.nn.softmax(tf.cast(attn_weights, tf.float32), axis=-1)
        attn_weights = tf.cast(attn_weights_float, attn_weights.dtype)
        dropout_layer = layers.Dropout(rate=self.dropout)
        attn_probs = dropout_layer(tf.cast(attn_weights_float, dtype=attn_weights.dtype))

        assert v is not None
        attn = tf.linalg.matmul(attn_probs, v) 
        assert attn.shape == (bsz * self.num_heads, tgt_len, self.head_dim)
        if attn.shape[1] == 1:
            # when sequence length == 1
            # the transpose is a no-op copy before reshape, thus unnecessary
            attn = tf.reshape(attn, [tgt_len, bsz, embed_dim])
        else:
            attn = tf.reshape(tf.transpose(attn, perm=[1, 0, 2]), 
                              [tgt_len, bsz, embed_dim])    
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = tf.transpose(
                                tf.cast(
                                    tf.reshape(attn_weights_float, [bsz, self.num_heads, tgt_len, src_len]), 
                                    attn.dtype
                                ), 
                                perm=[1, 0, 2, 3]
                            ) # shape: (H, B, T, T)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = tf.reduce_mean(attn_weights, axis=0)
        
        return attn, attn_weights