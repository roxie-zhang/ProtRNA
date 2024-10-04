import os
import numpy as np

from protrna.model import ProtRNA
from downstream_ss.model import SSPredictionModel


def load_pretrained_model(name="ProtRNA", path="./weights"):
    
    if name == "ProtRNA_pretrained":
        model = ProtRNA(num_layers=33,
                        embed_dim=1280,
                        attention_heads=20,
                        alphabet="ProtRNA",
                        token_dropout=False)
        model.trainable = False
        model(np.zeros((1, 512)), need_head_weights=True)  # build model
        file_path = os.path.join(path, f"{name}.h5")

    elif name == "ssHead_RF_bprna":
        model = SSPredictionModel()
        model(np.zeros((1, 10, 1280)), np.zeros((1, 10))) # build model
        file_path = os.path.join(path, f"{name}.h5")

    else:
        raise ValueError(f"Unknown model name: {name}")

    weights_url = "https://zenodo.org/records/13888473/files/"+name+".h5"
    if not os.path.exists(file_path):
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Downloading weights for {name}")
        os.system(f"wget {weights_url} -O {file_path}")
        
    print("Loading weights", name)
    model.load_weights(file_path)

    return model