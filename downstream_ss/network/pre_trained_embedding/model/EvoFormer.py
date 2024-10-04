import tensorflow as tf
from downstream_ss.network.pre_trained_embedding.model.AttentionModule import *

class Evoformer(tf.Module):
    def __init__(self, config, name='evoformer_iteration', global_config=None):
        super(Evoformer, self).__init__(name=name+"_"+str(global_config['iter']))
        self.config = config
        self.gc = global_config
        self.extra = self.config['mode'] == 'extra'
        with self.name_scope:
            self.msa_row = MSARowAttentionWithPairBias(self.config["msa_row_attention_with_pair_bias"], name='msa_row_attention_with_pair_bias', global_config=self.gc, dtype=DTYPE)
            self.outer_mod = OuterProductMean(self.config['outer_product_mean'], global_config=self.gc)
            self.msa_transition = Transition(self.config['msa_transition'], name='msa_transition', global_config=self.gc)

            self.triangle_multiplication_outgoing = TriangleMultiplication(self.config['triangle_multiplication_outgoing'], name='triangle_multiplication_outgoing', global_config=self.gc)
            self.triangle_multiplication_incoming = TriangleMultiplication(self.config['triangle_multiplication_incoming'], name='triangle_multiplication_incoming', global_config=self.gc)

            self.triangle_attention_starting_node = TriangleAttention(self.config['triangle_attention_starting_node'], name='triangle_attention_starting_node', global_config=self.gc)
            self.triangle_attention_ending_node = TriangleAttention(self.config['triangle_attention_ending_node'], name='triangle_attention_ending_node', global_config=self.gc)

            self.pair_transition = Transition(self.config['pair_transition'], name='pair_transition', global_config=self.gc)

    def msa2msa(self, msa_act, pair_act, training=False):
        
        msa_resi = self.msa_row(msa_act, pair_act)
        msa_act += msa_resi

        msa_resi = self.msa_transition(msa_act)
        msa_act += msa_resi

        return msa_act

    def pair2pair(self, msa_act, pair_act, training=False):
        
        pair_resi = self.outer_mod(msa_act)
        pair_act += pair_resi

        pair_resi = self.triangle_multiplication_outgoing(pair_act)
        pair_act += pair_resi

        pair_resi = self.triangle_multiplication_incoming(pair_act)
        pair_act += pair_resi

        pair_resi = self.triangle_attention_starting_node(pair_act)
        pair_act += pair_resi

        pair_resi = self.triangle_attention_ending_node(pair_act)
        pair_act += pair_resi

        pair_resi = self.pair_transition(pair_act)
        pair_act += pair_resi

        return pair_act

    def __call__(self, msa_act, pair_act, training=False):
        
        msa_act = self.msa2msa(msa_act, pair_act, training=training)
        pair_act = self.pair2pair(msa_act, pair_act, training=training)

        return msa_act, pair_act
