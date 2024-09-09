# ProtRNA

![graphical_abstract](https://github.com/user-attachments/assets/5835a565-268a-426e-9266-5a257e40df35)

## Abstract
Protein language models (PLM), such as the highly successful ESM-2, have proven to be particularly effective. However, language models designed for RNA continue to face challenges. A key question is: can the information derived from PLMs be harnessed and transferred to RNA? To investigate this, a model termed ProtRNA has been developed by cross-modality transfer learning strategy for addressing the challenges posed by RNA's limited and less conserved sequences. By leveraging the evolutionary and physicochemical information encoded in protein sequences, the ESM-2 model is adapted to processing "low-resource" RNA sequence data. The results show comparable or even superior performance in various RNA downstream tasks, with only 1/8 the trainable parameters and 1/6 the training data employed by other baseline RNA language models. This approach highlights the potential of cross-modality transfer learning in biological language models.

## Dependencies

`python>=3.9`
`numpy==1.26.1`
`tensorflow==2.14.0`

Note: Only tensorflow implementation is available for now.

## Usage

Model weights for pretrained ProtRNA model would be downloaded first time running the following code:

```python
from protrna.pretrained import load_model

model = load_model()
```

You may also manually download the weights by command line:
```
wget -O ProtRNA_pretrained.h5 'https://www.dropbox.com/scl/fi/lttcohy9ix1rbsynci72h/ProtRNA_pretrained.h5?rlkey=6t37r3tbb1p9j75nqejumcqqx&st=74eku09q&dl=1'
```

## Inference

```python
from protrna.pretrained import load_model

model = load_model()
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

