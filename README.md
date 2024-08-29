# ProtRNA

## Abstract
While protein language models (PLMs), such as the highly successful ESM-2, have proven particularly effective, language models designed for other modalities, like RNA, continue to face challenges. This study explores a key question: can the information encoded in protein language model be harnessed and transferred to other biological modalities such as RNA? To investigate this, we propose a cross-modality transfer learning strategy for BLMs, with a focus on foundational RNA language models. By leveraging the evolutionary and physicochemical information encoded in the state-of-the-art protein language model ESM-2, we adapt the model to "low-resource" RNA sequence data, addressing the challenges posed by RNA's limited and less conserved sequences. Our ProtRNA language model representations demonstrate comparable or superior performance in various RNA downstream tasks, achieving these results with only 1/8 the trainable parameters and 1/6 the training data employed by other baseline RNA language models. This approach highlights the potential of cross-modality transfer learning in BLMs, promoting broader accessibility and accelerating research and discovery in biological sequence modeling.

## Usage

Model weights for pretrained ProtRNA model would be downloaded first time running the following code:

```python
from pretrained import load_model

model = load_model()
```

You may also manually download the weights by command line:
```
wget -O ProtRNA_pretrained.h5 'https://www.dropbox.com/scl/fi/lttcohy9ix1rbsynci72h/ProtRNA_pretrained.h5?rlkey=6t37r3tbb1p9j75nqejumcqqx&st=74eku09q&dl=1'
```

## Inference
```python
from pretrained import load_model

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

