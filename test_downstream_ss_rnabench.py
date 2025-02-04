import numpy as np

import argparse

from pretrained import load_pretrained_model
from downstream_ss.metrics import sigmoid
from downstream_ss.RnaBench.benchmarks import RnaFoldingBenchmark


TASKS = ["inter", "intra", "bprna"]
MODEL_NAME = "ProtRNA_pretrained"


def main(args):

    task_name = args.task

    HEAD_NAME = f"ssHead_RF_rnabench_{task_name}"

    # Load pretrained models
    base_model = load_pretrained_model(name=MODEL_NAME)
    batch_converter = base_model.alphabet.get_batch_converter()

    ss_model = load_pretrained_model(name=HEAD_NAME)

    # =======================From Logits=======================

    # default, the best selected on val
    if task_name == "intra":
        THRESHOLD = 0.21
    elif task_name == "inter":
        THRESHOLD = 0.14
    elif task_name == "bprna":
        THRESHOLD = 0.29

    # =======================Validation=======================
    if args.val:
        if task_name == "bprna":
            folding_benchmark = RnaFoldingBenchmark(task="bprna_val")
        else:
            folding_benchmark = RnaFoldingBenchmark(task=f"{task_name}_family_valid")

        best_metrics = {}
        # Try different THRESHOLD values
        for threshold in np.arange(0.1, 0.4, 0.01):
    
            def protrna_prediction_wrapper(rna_folding_task):
                
                seq_tokens = batch_converter([''.join(rna_folding_task.sequence)])
                seq_results = base_model(seq_tokens, repr_layers=[33])
                logits = ss_model(seq_results['representations'][33])

                preds = sigmoid(logits) > THRESHOLD

                # Get upper triangular indices (excluding diagonal)
                tri_inds = np.triu_indices(preds.shape[0], k=1)

                # Create pred_pairs where preds[i, j] == 1, adding an extra [0] for pipeline wrapper formatting
                pred_pairs = [[tri_inds[0][i], tri_inds[1][i], 0] for i in range(len(tri_inds[0])) if preds[tri_inds[0][i], tri_inds[1][i]]]

                return pred_pairs

            # RnaBench will compute several metrics for your model predictions
            metrics = folding_benchmark(protrna_prediction_wrapper, save_results=False, algorithm_name=args.lm_model)
            best_metrics[threshold] = metrics["f1_score"]
        
        best_threshold = max(best_metrics, key=best_metrics.get)
        best_f1_score =  best_metrics[best_threshold]
        THRESHOLD = best_threshold
        print(f"best f1_score: {best_f1_score}")
        print(f"Finished Validation, picked best_threshold {best_threshold:.2f}.")

        
    # =======================Test=======================
    print(f"Start Testing, threshold {THRESHOLD}")

    if task_name == "bprna":
        folding_benchmark = RnaFoldingBenchmark(task="bprna_test")
    else:
        folding_benchmark = RnaFoldingBenchmark(task=f"{task_name}_family")

    def protrna_prediction_wrapper(rna_folding_task):

        seq_tokens = batch_converter([''.join(rna_folding_task.sequence)])
        seq_results = base_model(seq_tokens, repr_layers=[33])
        logits = ss_model(seq_results['representations'][33])

        preds = sigmoid(logits) > THRESHOLD

        # Get upper triangular indices (excluding diagonal)
        tri_inds = np.triu_indices(preds.shape[0], k=1)

        # Create pred_pairs where preds[i, j] == 1, adding an extra [0] for pipeline wrapper formatting
        pred_pairs = [[tri_inds[0][i], tri_inds[1][i], 0] for i in range(len(tri_inds[0])) if preds[tri_inds[0][i], tri_inds[1][i]]]

        return pred_pairs
    
    metrics = folding_benchmark(protrna_prediction_wrapper, save_results=True, algorithm_name=args.lm_model)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=TASKS, required=True
    )
    parser.add_argument(
        "--lm_model", type=str, default="ProtRNA",
    )
    parser.add_argument(
        "--val", action="store_true", default=False, 
    )
    args = parser.parse_args()
    main(args)