import tensorflow as tf
from downstream_ss.network.pre_trained_embedding.model.LinearModule import Linear
import numpy as np

import tensorflow.experimental.numpy as tnp
DTYPE = tf.float32
np_dtype = np.float32

class LayerNorm(tf.Module):
    def __init__(self, name=None, input_dim=1, create_scale=True, create_offset=True, global_config=None, dtype=tf.float32,
                 load=False):
        super(LayerNorm, self).__init__(name=name)
        with self.name_scope:
            self.gc = global_config
            if self.gc == None:
                self.gc = {'iter': 0}
            self.create_scale = create_scale
            self.create_offset = create_offset
            self.DTYPE = dtype
            if self.create_scale:
                self.scale = tf.Variable(tf.ones((input_dim,), dtype=self.DTYPE), name='scale')
            if self.create_offset:
                self.offset = tf.Variable(tf.zeros((input_dim,), dtype=self.DTYPE), name='offset')

    def __call__(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        eps = 1e-5
        if self.create_scale:
            inv = self.scale*tf.math.rsqrt(var + eps)
        else:
            inv = tf.math.rsqrt(var + eps)
        if self.create_offset:
            return inv*(inputs - mean) + self.offset
        else:
            return inv*(inputs - mean)

class Attention(tf.Module):
    def __init__(self, config, name="attention", global_config=None):
        super(Attention, self).__init__(name=name)
        self.gc = global_config
        self.config = config
        self.input_dim = self.config["input_dim"]
        self.key_dim = self.config["key_dim"]
        self.value_dim = self.config["value_dim"]
        self.num_head = self.config["num_head"]
        self.gating = self.config["gating"]
        self.output_dim = self.config["output_dim"]
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        self.DTYPE = DTYPE

        with self.name_scope:
            shape = [self.input_dim, self.num_head, self.key_dim]
            self.q_weights = tf.Variable(tf.ones(shape, dtype=self.DTYPE), name='query_w', dtype=self.DTYPE)
            self.k_weights = tf.Variable(tf.ones(shape, dtype=self.DTYPE), name='key_w'  , dtype=self.DTYPE)
            self.v_weights = tf.Variable(tf.ones(shape, dtype=self.DTYPE), name='value_w', dtype=self.DTYPE)

            o_shape = [self.num_head, self.value_dim, self.output_dim]
            self.o_weights = tf.Variable(tf.ones(o_shape, dtype=self.DTYPE), name='output_w', dtype=self.DTYPE)
            self.o_bias = tf.Variable(tf.zeros((self.output_dim,), dtype=self.DTYPE), name='output_b', dtype=self.DTYPE)

            if self.gating:
                g_shape = [self.input_dim, self.num_head, self.value_dim]
                self.gating_weights = tf.Variable(tf.zeros(g_shape, dtype=self.DTYPE), name='gating_w', dtype=self.DTYPE)
                self.gating_bias = tf.Variable(tf.ones([self.num_head, self.value_dim], dtype=self.DTYPE), name='gating_b', dtype=self.DTYPE)

        variables = [self.q_weights, self.k_weights, self.v_weights, self.o_weights, self.o_bias]
        variable_names = ['query_w', 'key_w', 'value_w', 'output_w', 'output_b']
        if self.gating:
            variables += [self.gating_weights, self.gating_bias]
            variable_names += ['gating_w', 'gating_b']

    def __call__(self, q_data, m_data, bias=None):

        q = tnp.einsum('bqa,ahc->bqhc', q_data, self.q_weights) / np.sqrt(self.key_dim)
        k = tnp.einsum('bka,ahc->bkhc', m_data, self.k_weights)
        v = tnp.einsum('bka,ahc->bkhc', m_data, self.v_weights)

        if bias is not None:
            bias = tf.expand_dims(bias, axis=0)
            logits = tnp.einsum('bqhc,bkhc->bhqk', q, k) + bias
        else:
            logits = tnp.einsum('bqhc,bkhc->bhqk', q, k)

        weights = tf.nn.softmax(logits, axis=-1)
        weighted_avg = tnp.einsum('bhqk,bkhc->bqhc', weights, v)
        if self.gating:
            gate_values = tnp.einsum('bqc, chv->bqhv', q_data, self.gating_weights) + self.gating_bias
            gate_values = tf.nn.sigmoid(gate_values)
            weighted_avg = weighted_avg*gate_values
        output = tnp.einsum('bqhc,hco->bqo', weighted_avg, self.o_weights) + self.o_bias

        return output

class MSARowAttentionWithPairBias(tf.Module):
    def __init__(self, config, name="msa_row_attention_with_pair_bias", global_config=None, dtype=tf.float32):
        super(MSARowAttentionWithPairBias, self).__init__(name=name)
        with self.name_scope:
            self.gc = global_config
            self.config = config
            self.input_dim = self.config['input_dim']
            self.pair_input_dim = self.config['pair_input_dim']
            self.DTYPE = dtype
            self.layer_norm = LayerNorm(name='query_norm', input_dim=self.input_dim, global_config=self.gc, dtype=self.DTYPE)
            self.pair_layer_norm = LayerNorm(name='feat_2d_norm', input_dim=self.pair_input_dim, global_config=self.gc, dtype=self.DTYPE)
            self.attn_mod = Attention(config, global_config=self.gc)

            self.num_head = self.config['num_head']
            shape = [self.pair_input_dim, self.num_head]
            self.weights = tf.Variable(tf.ones(shape, dtype=self.DTYPE), name='feat_2d_weights', dtype=self.DTYPE)

    def __call__(self, msa_act, pair_act, msa_mask=None):
        msa_act = tf.cast(msa_act, self.DTYPE)
        pair_act = tf.cast(pair_act, self.DTYPE)
        msa_act = self.layer_norm(msa_act)
        pair_act = self.pair_layer_norm(pair_act)

        nonbatched_bias = tnp.einsum('qkc,ch->hqk', pair_act, self.weights)
        msa_act = self.attn_mod(msa_act, msa_act, bias=nonbatched_bias)
        msa_act = tf.cast(msa_act, self.DTYPE)
        return msa_act

class TriangleAttention(tf.Module):
    def __init__(self, config, name='triangle_attention', global_config=None):
        super(TriangleAttention, self).__init__(name=name)
        with self.name_scope:
            self.DTYPE = DTYPE
            self.gc = global_config
            self.config = config
            self.input_dim = self.config["input_dim"]
            self.num_head = self.config["num_head"]
            self.layer_norm = LayerNorm(name='query_norm', input_dim=self.input_dim, global_config=self.gc)
            shape = [self.input_dim, self.num_head]

            self.weights = tf.Variable(tf.ones(shape, dtype=self.DTYPE), name='feat_2d_weights', dtype=self.DTYPE)

            self.attn_mod = Attention(config, global_config=self.gc)
            assert self.config['orientation'] in ['per_row', 'per_column']
            self.column_orientation = self.config["orientation"] == "per_column"

    def __call__(self, pair_act, pair_mask=None):

        if self.column_orientation:
            pair_act = tf.transpose(pair_act, [1,0,2])
        pair_act = self.layer_norm(pair_act)

        init_factor = 1./np.sqrt(pair_act.shape[-1])
        nonbatched_bias = tnp.einsum('qkc,ch->hqk', pair_act, self.weights)

        pair_act = tf.cast(pair_act, self.DTYPE)
        nonbatched_bias = tf.cast(nonbatched_bias, self.DTYPE)

        pair_act = self.attn_mod(pair_act, pair_act, bias=nonbatched_bias)

        pair_act = tf.cast(pair_act, tf.float32)

        if self.column_orientation:
            pair_act = tf.transpose(pair_act, [1,0,2])

        return pair_act

class TriangleMultiplication(tf.Module):
    def __init__(self, config, name='triangle_multiplication', global_config=None):
        super(TriangleMultiplication, self).__init__(name=name)
        with self.name_scope:
            self.gc = global_config
            self.config = config
            self.input_dim = self.config["input_dim"]
            self.layer_norm = LayerNorm(name='layer_norm_input', input_dim=self.input_dim, global_config=self.gc)
            self.num_intermediate_channel = self.config["num_intermediate_channel"]
            self.DTYPE = tf.float32

            self.layer_norm_center = LayerNorm(name='center_layer_norm', input_dim=self.num_intermediate_channel, global_config=self.gc)
            self.left_proj = Linear(self.num_intermediate_channel, num_input=self.input_dim, name='left_projection', global_config=self.gc, dtype=self.DTYPE)
            self.right_proj = Linear(self.num_intermediate_channel, num_input=self.input_dim, name='right_projection', global_config=self.gc, dtype=self.DTYPE)

            self.left_gate = Linear(self.num_intermediate_channel, num_input=self.input_dim, name='left_gate', initializer='gate', global_config=self.gc, dtype=self.DTYPE)
            self.right_gate = Linear(self.num_intermediate_channel, num_input=self.input_dim, name='right_gate', initializer='gate', global_config=self.gc, dtype=self.DTYPE)

            self.equation = config["equation"]

            self.output_channel = self.config["output_channel"]
            self.output = Linear(self.output_channel, num_input=self.num_intermediate_channel, name='output_projection', global_config=self.gc, dtype=self.DTYPE)

            self.gate = Linear(self.output_channel, num_input=self.num_intermediate_channel, name='gating_linear', initializer='gate', global_config=self.gc, dtype=self.DTYPE)

    def __call__(self, act, mask=None):
        act = self.layer_norm(act)
        act = tf.cast(act, self.DTYPE)
        input_act = act

        left_proj_act = self.left_proj(act)

        right_proj_act = self.right_proj(act)

        left_gate_values = tf.nn.sigmoid(self.left_gate(act))

        right_gate_values = tf.nn.sigmoid(self.right_gate(act))

        left_proj_act *= left_gate_values

        right_proj_act *= right_gate_values

        act = tnp.einsum(self.equation, left_proj_act, right_proj_act)

        act = tf.cast(act, tf.float32)
        act = self.layer_norm_center(act)
        act = tf.cast(act, self.DTYPE)
        act = self.output(act)

        gate_values = tf.nn.sigmoid(self.gate(input_act))

        act *= gate_values
        act = tf.cast(act, tf.float32)

        return act

class OuterProductMean(tf.Module):

    def __init__(self, config, name='outer_product_mean', global_config=None):
        super(OuterProductMean, self).__init__(name=name)
        with self.name_scope:

            self.gc = global_config
            self.config = config

            self.num_input_channel = self.config["num_input_channel"]
            self.num_outer_channel = self.config["num_outer_channel"]
            self.num_output_channel = self.config["num_output_channel"]

            self.DTYPE = tf.float32

            self.left_proj = Linear(self.num_outer_channel, num_input=self.num_input_channel, name="left_projection", global_config=self.gc, dtype=self.DTYPE)

            self.right_proj = Linear(self.num_outer_channel, num_input=self.num_input_channel, name="right_projection", global_config=self.gc, dtype=self.DTYPE)

            o_shape = [self.num_outer_channel, self.num_outer_channel, self.num_output_channel]
            self.output_w = tf.Variable(tf.ones(o_shape, dtype=self.DTYPE), name="output_w", dtype=self.DTYPE)
            self.output_b = tf.Variable(tf.zeros((self.num_output_channel,), dtype=self.DTYPE), name="output_b", dtype=self.DTYPE)

            self.layer_norm = LayerNorm(name='layer_norm_input', input_dim=self.num_input_channel, global_config=self.gc)

    def __call__(self, act, mask=None):

        norm = act.shape[-3]
        act = self.layer_norm(act)

        left_act = self.left_proj(act)
        right_act = self.right_proj(act)

        left_act = tf.transpose(left_act, [0, 2, 1])
        act = tnp.einsum('acb,ade->dceb', left_act, right_act)
        act = tnp.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        act = tf.transpose(act, [1, 0, 2])

        act /= norm + 1e-3

        return act

class Transition(tf.Module):
    def __init__(self, config, name='transition_block', global_config=None):
        super(Transition, self).__init__(name=name)
        with self.name_scope:
            self.gc = global_config
            self.config = config
            self.input_dim = self.config["input_dim"]
            self.DTYPE = DTYPE

            self.num_intermediate = self.config["num_intermediate_factor"]*self.input_dim

            self.layer_norm = LayerNorm(name='input_layer_norm', input_dim=self.input_dim, global_config=self.gc, dtype=self.DTYPE)

            self.transition1 = Linear(self.num_intermediate, num_input=self.input_dim, name='transition1', initializer='relu', global_config=self.gc, dtype=self.DTYPE)

            self.transition2 = Linear(self.input_dim, num_input=self.num_intermediate, name='transition2', initializer='final', global_config=self.gc, dtype=self.DTYPE)

    def __call__(self, act):
        act = tf.cast(act, self.DTYPE)
        act = self.layer_norm(act)

        act = tf.nn.relu(self.transition1(act))

        act = self.transition2(act)
        return act
