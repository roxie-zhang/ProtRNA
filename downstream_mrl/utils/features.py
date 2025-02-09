import os
import pandas as pd

import numpy as np
import tensorflow as tf


def prepare_features(base_model, batch_converter, data_root: str, feature_path: str, dataset: str):
    
    dataset_path = f"{data_root}/split.csv.gz"
    data = pd.read_csv(dataset_path)

    # Assuming the column with the label splits is named 'split' and the test split is labeled 'test'
    test_idcs = data[data['split'] == 'test'].index.tolist()
    print(f"{len(test_idcs)} rows in test dataset")

    os.makedirs(feature_path, exist_ok=True)
    for i in test_idcs:
        seq = data.loc[i, 'utr']
        seq_tokens = batch_converter([seq])
        seq_results = base_model(seq_tokens, repr_layers=[33])
        reprs = tf.squeeze(seq_results['representations'][33])
        save_path = f"{feature_path}/{i}.npy"
        np.save(save_path, reprs)

        if i % 100 == 0:
            print(f"{i}-th sequence done")
    
    print(f"all sequences for {dataset} saved in {feature_path}")