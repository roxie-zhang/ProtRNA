import os
import numpy as np

from model import ProtRNA


def load_model(name="ProtRNA_pretrained.h5", path="./weights"):
    model = ProtRNA(num_layers=33,
                    embed_dim=1280,
                    attention_heads=20,
                    alphabet="ProtRNA",
                    token_dropout=False)
    model.trainable = False
    model(np.zeros((1, 512)), need_head_weights=True)

    file_path = os.path.join(path, name)

    if not os.path.exists(file_path):
        if not os.path.exists(path):
            os.makedirs(path)
        print("Downloading weights")
        if name == "ProtRNA_pretrained.h5":
            os.system(f"wget 'https://www.dropbox.com/scl/fi/lttcohy9ix1rbsynci72h/ProtRNA_pretrained.h5?rlkey=6t37r3tbb1p9j75nqejumcqqx&st=74eku09q&dl=1' -O {file_path}")
        else:
            raise ValueError(f"Unknown model name: {name}")
        
    print("Loading weights", name)
    model.load_weights(file_path)

    return model