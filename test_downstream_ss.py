import os
import pickle

import numpy as np
from tqdm import tqdm

from pretrained import load_pretrained_model
from downstream_ss.metrics import *


THRESHOLD = 0.25 # selected on VL0 during training
MODEL_NAME = "ProtRNA_pretrained"
HEAD_NAME = "ssHead_RF_bprna"

def print_avg_metric(metric_name, metric_list):
    print(f"{metric_name}:", sum(metric_list)/len(metric_list))


if __name__ == '__main__':

    # Load pretrained models
    base_model = load_pretrained_model(name=MODEL_NAME)
    batch_converter = base_model.alphabet.get_batch_converter()

    ss_model = load_pretrained_model(name=HEAD_NAME)

    # Load dataset
    filename = 'TS0'
    data_dir = './downstream_ss/dataset'

    print('processing', filename)
    file_path = os.path.join(data_dir, filename+'.pickle')
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print('dataset length:', len(dataset))

    # Calculate metrics
    metrics = {
        'f1': [],
        'MCC': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'f1_shifted': []
    }

    for row in tqdm(dataset):
        seq_len = row["seq_len"]
        pairs = row["pairs"]
        target = np.zeros((seq_len, seq_len))
        target[tuple(zip(*pairs))] = 1

        seq_tokens = batch_converter([row["seq"]])
        seq_results = base_model(seq_tokens, repr_layers=[33])
        pred = ss_model(seq_results['representations'][33])
        pred = sigmoid(pred) > THRESHOLD

        metrics['f1'].append(F1_score(pred, target))
        metrics['MCC'].append(MCC(pred, target))
        metrics['precision'].append(precision(pred, target))
        metrics['recall'].append(recall(pred, target))
        metrics['accuracy'].append(accuracy(pred, target))
        metrics['f1_shifted'].append(ss_f1(target, pred))

    print(f'{MODEL_NAME} {HEAD_NAME} average metrics on {filename}:')
    for metric_name, metric_list in metrics.items():
        print_avg_metric(metric_name, metric_list)
        
    