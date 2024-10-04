import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def F1_score(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return f1_score(true_a[tril_index], opt_state[tril_index])

def accuracy(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return accuracy_score(true_a[tril_index], opt_state[tril_index])

def precision(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return precision_score(true_a[tril_index], opt_state[tril_index])

def recall(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return recall_score(true_a[tril_index], opt_state[tril_index])

def MCC(opt_state, true_a):
    tril_index = np.tril_indices(len(opt_state),k=-1)
    return matthews_corrcoef(true_a[tril_index], opt_state[tril_index])

def _relax_ss(ss_mat: np.array) -> np.array:
    # Pad secondary structure (because of cyclical rolling)
    ss_mat = np.pad(ss_mat, ((1, 1), (1, 1)), mode='constant')

    # Create relaxed pairs matrix
    relax_pairs = \
            np.roll(ss_mat, shift=1, axis=-1) + np.roll(ss_mat, shift=-1, axis=-1) +\
            np.roll(ss_mat, shift=1, axis=-2) + np.roll(ss_mat, shift=-1, axis=-2)

    # Add relaxed pairs into original matrix
    relaxed_ss = ss_mat + relax_pairs

    # Ignore cyclical shift and clip values
    relaxed_ss = relaxed_ss[..., 1: -1, 1: -1]
    relaxed_ss = np.clip(relaxed_ss, 0, 1)

    return relaxed_ss

def ss_recall(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    if allow_flexible_pairings:
        pred_ss = _relax_ss(pred_ss)
    
    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)

    return recall_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)

def ss_precision(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    if allow_flexible_pairings:
        target_ss = _relax_ss(target_ss)
    
    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)

    return precision_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)

EPSILON = 1e-5

def ss_f1(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    precision = ss_precision(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)
    recall = ss_recall(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)

    # Prevent division with 0.0
    if precision + recall < EPSILON:
        return 0.0

    return (2 * precision * recall) / (precision + recall)