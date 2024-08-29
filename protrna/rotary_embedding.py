from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, initializers


def rotate_half(x):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : tf.shape(x)[-2], :]   # cos[:, :, : tf.shape(x)[-2], :] in huggingface code
    sin = sin[:, : tf.shape(x)[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(layers.Layer):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, name=None):
        super().__init__(name=name)
        self.dim = dim

    def build(self, input_shape):
        super().build(input_shape)
        self.inv_freq = self.add_weight(
            "inv_freq", 
            shape=(self.dim // 2,),
            dtype=tf.float32,
            initializer=initializers.TruncatedNormal(stddev=1.0),
            trainable=False,
        )
        self.inv_freq.assign(
            1.0 / (10000 ** (tf.range(0, self.dim, 2.0, dtype=tf.float32) / self.dim))
        )

    def _update_cos_sin_tables(self, x, seq_dimension=1):     # seq_dimension=2 in huggingface code
        seq_len = tf.shape(x)[seq_dimension]

        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)[None, :, :]  # [None, None, :, :] in huggingface code

        return tf.cos(emb), tf.sin(emb)

    def call(self, q: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        cos_emb, sin_emb = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, cos_emb, sin_emb),
            apply_rotary_pos_emb(k, cos_emb, sin_emb),
        )
