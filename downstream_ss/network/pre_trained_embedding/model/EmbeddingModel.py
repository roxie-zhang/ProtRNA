import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from downstream_ss.network.pre_trained_embedding.model.LinearModule import Linear

class Embedding(tf.Module):
    def __init__(self, config, name='Embedding'):
        super(Embedding, self).__init__(name=name)
        
        c_m = config['n_1d_feat']
        c_p = config['n_2d_feat']
        self.max_rd = config['max_relative_distance']
        
        self.pre_linear_1d = Linear(c_m, num_input=1280, name='preprocess_1d')

        self.pre_linear_left = Linear(c_p, num_input=1280, name='left_single')
        self.pre_linear_right = Linear(c_p, num_input=1280, name='right_single')

        self.pre_linear_pair = Linear(c_p, num_input=65, name='pair_activiations')

    def __call__(self, inp_1d, residue_index):
        
        target_feat = inp_1d[0]
        
        preprocess_1d = self.pre_linear_1d(target_feat)
        msa_activations = tnp.expand_dims(preprocess_1d, axis=0)

        left_single = self.pre_linear_left(target_feat)
        right_single = self.pre_linear_right(target_feat)
        pair_activations = left_single[:, None] + right_single[None]

        pos = residue_index
        offset = pos[:, None] - pos[None, :]
        rel_pos = tf.one_hot(
            tnp.clip(offset + self.max_rd, a_min=0, a_max=2 * self.max_rd),
            2 * self.max_rd + 1)
        pair_activations += self.pre_linear_pair(rel_pos)

        return msa_activations, pair_activations
