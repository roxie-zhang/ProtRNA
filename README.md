# ProtRNA: A Protein-derived RNA Language Model

![graphical_abstract](https://github.com/user-attachments/assets/5835a565-268a-426e-9266-5a257e40df35)

## Abstract
Protein language models (PLM), such as the highly successful ESM-2, have proven to be particularly effective. However, language models designed for RNA continue to face challenges. A key question is: can the information derived from PLMs be harnessed and transferred to RNA? To investigate this, a model termed ProtRNA has been developed by cross-modality transfer learning strategy for addressing the challenges posed by RNA's limited and less conserved sequences. By leveraging the evolutionary and physicochemical information encoded in protein sequences, the ESM-2 model is adapted to processing "low-resource" RNA sequence data. The results show comparable or even superior performance in various RNA downstream tasks, with only 1/8 the trainable parameters and 1/6 the training data employed by other baseline RNA language models. This approach highlights the potential of cross-modality transfer learning in biological language models.

## Installation

First create a conda environment with `python=3.9`
```
conda create --name protrna python=3.9
conda activate protrna
```

After cloning the repo, you may install the requirements by
```
pip install -r requirements.txt
```

where `numpy==1.26.1` and `tensorflow==2.14.0` are sufficient for running the base model inference, and `tqdm` and 
`scikit-learn` are additional requirements for the downstream evaluation pipeline.

To evaluate downstream tasks performance on secondary structure prediction or protein-RNA interaction prediction, you may refer to [RnaBench](https://github.com/automl/RnaBench) and [PrismNet](https://github.com/kuixu/PrismNet) for dependencies installation.

## Inference

Model weights for the ProtRNA model would be downloaded first time running the following code:

```python
from pretrained import load_pretrained_model

model = load_pretrained_model(name='ProtRNA_pretrained')
```

To generate ProtRNA representations of RNA sequences, run:
```python
from pretrained import load_pretrained_model

model = load_pretrained_model(name='ProtRNA_pretrained')
batch_converter = model.alphabet.get_batch_converter()

seqs = [
        "CAUAUCGUGAGAUGUGGGCGAGAAGAAGGGAUAGCGAAAUCGUAGCCCUACGGACAGAAACCUGAUAAUAAGGCGUGCAUGGCGGGUAAGUUGGCUUAAAGCAACGAAGCCCUAAAGGUAGCCGUAACCUAUGUGCGUAAAUUAGGAGGGUAGACGAGGAAAGAACACG",
        "AAAAGUUAAAAAUGAUAUUCCCGAAAGGAUGCACCAUGUGUAGAUGCCUUGUAACCGGAAUUGAAUGGGGGAAAAAGAAAUG",
        "AAGGUCGAGUGAUGAGCAAAAAUGCAUAGUUCAGAUGAUCAAACCCUAGUGGUUAUGAUUACUUUGAAUAAAUAGUCUUUCGCUCCUAACUGACGGCCUU",
]

seq_tokens = batch_converter(seqs)
results = model(seq_tokens, repr_layers=[33])

print("ProtRNA embeddings:", results["representations"][33])
```

## Evaluation

For detailed descriptions and manual download links for the downstream task datasets and head weights, please visit https://zenodo.org/records/14795554. 

Before running the downstream evaluation tasks, download the required datasets and head weights by executing:

```bash
./prepare_data.sh
```

### Secondary Structure prediction

For the secondary structure prediction task on bpRNA-1m, evaluated with [RnaBench](https://github.com/automl/RnaBench) metrics, run:
```
python test_downstream_ss_rnabench.py --task bprna
```
The average performance metrics on TS0 will be reported.

Note: In a previous version of the paper, this task is evaluated using our own implemented metrics. To run that version, use:
```
python test_downstream_ss.py
```

### Protein-RNA Interaction prediction

The protein-RNA interaction datasets and prediction head architecture are sourced from [PrismNet](https://github.com/kuixu/PrismNet). 

To evaluate a specific protein, e.g. `WTAP_Hela`, run:
```
./test_downstream_rbp.sh WTAP_Hela
```   

### Mean Ribosome Loading prediction

The mean ribosome loading prediction head is sourced from [RiNALMo](https://github.com/lbcb-sci/RiNALMo).

To evaluate on one of the test sets, run:
```
 python test_downstream_mrl.py --test random7600
```
