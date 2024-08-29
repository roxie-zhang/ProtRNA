from typing import Union
import tensorflow as tf
from tensorflow.keras import layers, Model

from protrna.data import Alphabet
from protrna.modules import RobertaLMHead, TransformerLayer


class ProtRNA(Model):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[Alphabet, str] = "ProtRNA",
        token_dropout: bool = False,
    ):
        super().__init__(name='ProtRNA')
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_tokens = layers.Embedding(self.alphabet_size, self.embed_dim, name="embedding")
        self.embed_tokens(tf.constant([[0]])) #initialize embedding layer
        self.layer_ =[
                TransformerLayer(
                    name=f"layer_._{i}",
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    add_bias_kv=False,
                    use_rotary_embeddings=True,
                )
                for i in range(self.num_layers)
            ]

        self.emb_layer_norm_after = layers.LayerNormalization(epsilon=1e-5, name="emb_layer_norm_after")

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            decoder=self.embed_tokens.weights[0],
        )
    
    def call(self, tokens, repr_layers=[], need_head_weights=False):
        assert len(tf.shape(tokens)) == 2
        padding_mask = tf.math.equal(tokens, self.padding_idx)  # B, T

        x = self.embed_tokens(tokens)

        if self.token_dropout:
            x = tf.where((tokens == self.mask_idx)[:, :, None], 0.0, x)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = tf.reduce_sum(tf.cast(~padding_mask, x.dtype), axis=-1)
            masked_tokens = tokens == self.mask_idx
            mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1)/ src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - tf.cast(tf.expand_dims(padding_mask, -1), dtype=x.dtype))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = tf.transpose(x, perm=[1, 0, 2])

        # Encoder Attention Layers
        for layer_idx, layer in enumerate(self.layer_.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = tf.transpose(x, perm=[1, 0, 2])
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(tf.transpose(attn, perm=[1, 0, 2, 3]))

        x = self.emb_layer_norm_after(x)
        x = tf.transpose(x, perm=[1, 0, 2]) # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        # LM Head
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = tf.stack(attn_weights, 1)
            if not tf.reduce_any(padding_mask):
                attention_mask = 1 - tf.cast(padding_mask, dtype=attentions.dtype)
                attention_mask = tf.expand_dims(attention_mask, 1) * tf.expand_dims(attention_mask, 2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions

        return result

    

    
