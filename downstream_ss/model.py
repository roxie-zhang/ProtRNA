import tensorflow as tf
from tensorflow.keras import Model

from downstream_ss.network.trackable_layer import TrackableLayer
import downstream_ss.network.pre_trained_embedding.model.EmbeddingModel as Pre_MSA_emb
from downstream_ss.network.pre_trained_embedding import Settings
import downstream_ss.network.pre_trained_embedding.model.EvoFormer as EvoFormer

class AAEmbedding(TrackableLayer):
    def __init__(self, config):
        super(AAEmbedding, self).__init__()
        self.config = config
        self.emb = Pre_MSA_emb.Embedding(self.config)

    def call(self, inp_1d, residue_index):
        return self.emb(inp_1d, residue_index)

class AFEvoformer(TrackableLayer):
    def __init__(self, config, name_layer, name='evoformer_iteration', global_config=None):
        super(AFEvoformer, self).__init__(name=name_layer+"_"+str(global_config['iter']))
        self.config = config
        self.global_config = global_config
        self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
    def call(self, msa, pair, training=False):
        return self.evo_iteration(msa, pair, training=training)

class AFEvoformerEnsemble(TrackableLayer):
    def __init__(self, config, name_layer, iter_layer, name='evoformer_iteration', iters=None):
        super(AFEvoformerEnsemble, self).__init__(name=name_layer+"_"+str(iter_layer))
        self.config = config
        self.evo_iterations = []
        for i in range(len(iters)):
            global_config = {'iter': iters[i]}
            self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
            self.evo_iterations.append(self.evo_iteration)

    def call(self, msa, pair, training=False):
        for i in range(len(self.evo_iterations)):
            msa, pair = self.evo_iterations[i](msa, pair, training=training)
        return msa, pair

class SSPredictionModel(Model):
    def __init__(
            self, 
            ):
        super().__init__()
        self.pred_head = tf.keras.layers.Conv2D(filters=1, kernel_size=5, padding='SAME')
        
        self.model_params = {}
        self.model_params["n_1d_feat"] = 256
        self.model_params["n_2d_feat"] = 128
        self.model_params["max_relative_distance"] = 32
        self.model_params["evofomer_config"] = Settings.CONFIG
        
        self.pre_msa_emb = AAEmbedding(self.model_params)

        self.evoformers = []
        for i in range(3):
            self.evoformers.append(AFEvoformerEnsemble(self.model_params["evofomer_config"]["evoformer"],
                                                       name_layer='evoformer_ensemble',
                                                       iter_layer=i,
                                                       iters=[4*i, 4*i+1, 4*i+2, 4*i+3]))
        
    def call(self, inputs, training=False):
        representation = inputs

        inp_1d = representation[:, 1:-1, :]
        L = tf.shape(inp_1d)[1]
        residue_index = tf.range(0, L)

        assert inp_1d.shape == (1, L, 1280)
        
        f_1d, f_2d = self.pre_msa_emb(inp_1d, residue_index) # (1, L, 256) (L, L, 128)
        for i in range(len(self.evoformers)):
            f_1d, f_2d = self.evoformers[i](f_1d, f_2d, training=training)
        
        logits = self.pred_head(tf.expand_dims(f_2d, axis=0)) # assumes batch_size is 1
        sym_x = 0.5 * (logits + tf.transpose(logits, perm=[0, 2, 1, 3]))
        
        return tf.squeeze(sym_x) # (L, L)

            