import torch
import numpy as np

# from grakel import Graph
# from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath
from scipy import signal

from downstream_ss.RnaBench.lib.utils import db2mat, pairs2mat


def get_metric(m):
    """
    Returns performance measure functions.
    """
    if m == 'f1_score':
        return f1
    elif m == 'mcc':
        return mcc
    # elif m == 'wl':
    #     return weisfeiler_lehman_score_from_string
    elif m == 'recall':
        return recall
    elif m == 'precision':
        return precision
    elif m == 'specificity':
        return specificity
    elif m == 'solved':
        return solved_from_string
    elif m == 'f1_shifted':
        return evaluate_shifted_f1


# def get_general_dist(m):
#     if m == 'wl':
#         return weisfeiler_lehman_score_from_string


class BaseMetrics():
    """
    Base class for all metrics.
    """
    def __init__(self, metrics):
        for att, value in metrics.items():
            setattr(self, att, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        for att in vars(self):
            yield att, self[att]

    def items(self):
        for att in self:
            yield att, self[att]

    def values(self):
        for att in self:
            yield self[att]

    def keys(self):
        for att in self:
            yield att

    def __call__(self, row, *args, **kwargs):
        return self.evaluate(row, *args, **kwargs)


class GoalDirectedMetrics(BaseMetrics):
    """
    Evaluated metrics that can be used for goal-oriented design.
    """

    def __init__(self, metrics):
        """
        Initialize desired metrics.
        """
        metrics = {k: get_metric(k) for k in metrics}
        super().__init__(metrics)

    def evaluate(self,
                 row,
                 desired_gc=None,
                 gc_tolerance=0.01,
                 constrained=False,
                 ):
        """
        Evaluate all metrics.
        Called on a datafram row.

        """
        if isinstance(row['predicted_sequence'], float) or isinstance(row['predicted_pairs'], float):
            return {m: np.nan for m, _ in self}
        else:
            results = {}

            sequence = row['sequence']
            pred_sequence = row['predicted_sequence']

            if desired_gc is not None:
                # gc = row['gc_content']
                gc = desired_gc
                min_gc = gc - gc_tolerance
                max_gc = gc + gc_tolerance
                pred_gc = (''.join(pred_sequence).count('G') + ''.join(pred_sequence).count('C')) / len(pred_sequence)
                results['valid_gc_content'] = min_gc < pred_gc < max_gc
                results['gc_score'] = 1 - abs(gc - pred_gc)

            if constrained:
                true_seq = np.asarray(sequence)
                pred_seq = np.asarray(pred_sequence)
                d = (true_seq != pred_seq).astype(np.int8)
                ignore = (true_seq != 'N').astype(np.int8)
                distance = np.sum(d * ignore)
                results['valid_sequence_constraints'] = distance == 0
                results['sequence_distance_score'] = 1 - (distance / len(sequence))

            true_mat = pairs2mat(row['pairs'], length=row['length'])
            pred_mat = pairs2mat(row['predicted_pairs'], length=row['length'])

            tp = tp_from_matrices(pred_mat, true_mat)
            fp = get_fp(pred_mat, tp)
            fn = get_fn(true_mat, tp)
            tn = tn_from_matrices(pred_mat, true_mat)

            for name, metric in self:
                if name == 'gc':
                    continue
                # if name == 'wl':
                #     results['weisfeiler_lehman'] = graph_distance_score_from_matrices(
                #         pred_mat,
                #         true_mat,
                #         kernel='WeisfeilerLehman'
                #     )
                elif name == 'solved':
                    results['solved'] = solved_from_mat(pred_mat, true_mat)
                elif name == 'f1_shifted':
                    results['f1_shifted'] = evaluate_shifted_f1(pred_mat, true_mat)
                else:
                    results[name] = metric(tp, fp, tn, fn)
        # results['time'] = row['time']
        return results


# is tested
def f1_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs, length=len(pred))
    return f1_score_from_matrices(pred_mat, true_mat)


def mcc_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return mcc_from_matrices(pred_mat, true_mat)


def recall_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return recall_score_from_matrices(pred_mat, true_mat)


def specificity_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return specificity_score_from_matrices(pred_mat, true_mat)


def precision_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return precision_score_from_matrices(pred_mat, true_mat)


def solved_from_string(pred, true):
    return int(pred == true)


def non_correct_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    tp = tp_from_matrices(pred_mat, true_mat)
    return non_correct(tp)


# def weisfeiler_lehman_score_from_string(pred, true_pairs, kernel='WeisfeilerLehman'):
#     pred_mat = db2mat(pred)
#     true_mat = pairs2mat(true_pairs)
#     return graph_distance_score_from_matrices(pred_mat, true_mat, kernel=kernel)


################################################################################
# from e2efold
################################################################################
# we first apply a kernel to the ground truth a
# then we multiple the kernel with the prediction, to get the TP allows shift
# then we compute f1
# we unify the input all as the symmetric matrix with 0 and 1, 1 represents pair
def evaluate_shifted_f1(pred_a, true_a):
    pred_a = torch.tensor(pred_a)
    true_a = torch.tensor(true_a)

    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [0.0, 1.0, 0.0]])
    pred_a_filtered = signal.convolve2d(pred_a, kernel, 'same')
    fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered)) == 1)[0])
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    tp = true_p - fn
    fp = pred_p - tp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    return f1_score.item()



################################################################################
# For Matrices
################################################################################

################################################################################
# Structure
################################################################################

def f1_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    fn = get_fn(true, tp)
    return f1(tp, fp, None, fn)


def mcc_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    fn = get_fn(true, tp)
    tn = tn_from_matrices(pred, true)
    return mcc(tp, tn, fp, fn)


def recall_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fn = get_fn(true, tp)
    return recall(tp, fn)


def specificity_score_from_matrices(pred, true):
    tn = tn_from_matrices(pred, true)
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    return specificity(tn, fp)


def precision_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    return precision(tp, fp)


def solved_from_mat(pred, true):
    solved = np.all(np.equal(true, pred)).astype(int)
    return solved


# def graph_distance_score_from_pairs(pred, true, kernel='WeisfeilerLehman', node_labels=None):
#     if isinstance(true[0], list):
#         true = np.concatenate(true)
#         pred = np.concatenate(pred)
#     length = max(true.max(), pred.max()) + 1
#     true_mat = pairs2mat(true, length=length, no_pk=False)
#     pred_mat = pairs2mat(pred, length=length, no_pk=False)
#     return graph_distance_score_from_matrices(pred_mat, true_mat, kernel, node_labels=node_labels)


# def graph_distance_score_from_matrices(pred, true, kernel, node_labels=None):
#     pred_graph = mat2graph(pred, node_labels=node_labels)
#     true_graph = mat2graph(true, node_labels=node_labels)
#     kernel = get_graph_kernel(kernel=kernel)
#     kernel.fit_transform([true_graph])
#     distance_score = kernel.transform([pred_graph])  # TODO: Check output, might be list or list of lists

#     return distance_score[0][0]


################################################################################
# Helpers
################################################################################

# def get_graph_kernel(kernel, n_iter=5, normalize=True):
#     if kernel == 'WeisfeilerLehman':
#         return WeisfeilerLehman(n_iter=n_iter,
#                                 normalize=normalize,
#                                 base_graph_kernel=VertexHistogram)
#     elif kernel == 'WeisfeilerLehmanOptimalAssignment':
#         return WeisfeilerLehmanOptimalAssignment(n_iter=n_iter,
#                                                  normalize=normalize)
#     elif kernel == 'ShortestPath':
#         return ShortestPath(normalize=normalize)


# def mat2graph(matrix, node_labels=None):
#     if node_labels is not None:
#         graph = Graph(initialization_object=matrix.astype(int),
#                       node_labels=node_labels)  # TODO: Think about if we need to label the nodes differenty
#     else:
#         graph = Graph(initialization_object=matrix.astype(int),
#                       node_labels={s: str(s) for s in
#                                    range(
#                                        matrix.shape[0])})  # TODO: Think about if we need to label the nodes differenty

#     return graph

# is tested
def f1(tp, fp, tn, fn):
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1_score

# is tested
def recall(tp, fp, tn, fn):
    recall = tp / (tp + fn + 1e-8)
    return recall

# is tested
def specificity(tp, fp, tn, fn):
    specificity = tn / (tn + fp + 1e-8)
    return specificity

# is tested
def precision(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-8)
    return precision

# is tested
def mcc(tp, fp, tn, fn):
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    return mcc


def non_correct(tp, fp, tn, fn):
    non_correct = (tp == 0).astype(int)
    return non_correct

# is tested
def tp_from_matrices(pred, true):
    tp = np.logical_and(pred, true).sum()
    return tp

# is tested
def tn_from_matrices(pred, true):
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()
    return tn

# is tested
def get_fp(pred, tp):
    fp = pred.sum() - tp
    return fp

# is tested
def get_fn(true, tp):
    fn = true.sum() - tp
    return fn


def to_score(metric):
    return 1 - metric
