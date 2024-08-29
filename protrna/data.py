import itertools
from typing import Sequence, Tuple, List, Union

from protrna.constants import proteinseq_toks, rnaseq_toks

import tensorflow as tf


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):   ## required for esm-1b
            self.all_toks.append(f"<null_{i + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
        elif name in ("RNA-ESM", "rna_esm"):
            standard_toks = rnaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
        elif name == "ProtRNA":
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ["<mask>"] + rnaseq_toks["toks"]
            prepend_bos = True
            append_eos = True
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`): The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

    def get_batch_converter(self, is_rna=True):
        return BatchConverter(self, is_rna)
    

def _to_rna_vocab(sequence):
    """
    As the Alphabet of the model supports both protein and rna seqs,
    input transformation is required to distinguish between the two modalities.
    """
    sequence = sequence.lower()
    mapping = {'a': 'a', 'g': 'g', 't': 'u', 'c': 'c', 'u': 'u'}
    # Convert according to the mapping, default to 'x' if not in the mapping
    return ''.join(mapping.get(char, 'x') for char in sequence)


class BatchConverter(object):
    def __init__(self, alphabet, is_rna):
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.is_rna = is_rna
    
    def __call__(self, seq_strs_batch: Union[List[str], Tuple[str]], allowed_len=512) -> tf.Tensor:
        """
        Processes a batch of sequence strings.

        Parameters:
        seq_strs_batch (List[str]): An unprocessed batch of sequence strings.

        Returns:
        tf.Tensor: A processed batch of tensors. This involves adding special tokens, 
                converting to tensors, and padding to the maximum length sequence in the batch.
        """
        batch_size = len(seq_strs_batch)
        if self.is_rna:
            seq_strs_batch = [_to_rna_vocab(seq_str) for seq_str in seq_strs_batch]
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_strs_batch]

        max_seqlen = min(allowed_len - int(self.alphabet.prepend_bos) - int(self.alphabet.append_eos),
                         max(len(seq_encoded) for seq_encoded in seq_encoded_list))

        tokens_list = []
        for seq_encoded in seq_encoded_list:
            seq_tensor = tf.convert_to_tensor(seq_encoded[:max_seqlen], dtype=tf.int32)
            if self.alphabet.prepend_bos:
                seq_tensor = tf.concat([[self.alphabet.cls_idx], seq_tensor], axis=0)
            if self.alphabet.append_eos:
                seq_tensor = tf.concat([seq_tensor, [self.alphabet.eos_idx]], axis=0)
            tokens_list.append(seq_tensor)

        # Pad all sequences to the same length
        max_len = max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
        tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens_list, maxlen=max_len, padding='post', value=self.alphabet.padding_idx)
        
        assert tokens.shape == (batch_size, max_len)

        return tokens


def from_fasta(data_list, max_len=4096, with_label=False, split_label=False):
    """
    Returns a list of sequences from a fasta file.

    Parameters:
    max_len: Sequences longer than max_len are discarded.
    split_label: Only preserve seqname as label, for cases such as ">RF00001_M28193_1_1-119 5S_rRNA".
    """
    sequence_strs = []
    buf = []
    sequence_labels = []
    cur_seq_label = None

    def _flush_current_seq(with_label):
        nonlocal cur_seq_label, buf
        if with_label:
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []
        else:
            if buf:
                seq = "".join(buf)
                if len(seq) <= max_len:
                    sequence_strs.append("".join(buf))
                buf = []      

    with open(data_list, "r") as file:
        for line in file:
            if line.startswith(">"): # header line
                _flush_current_seq(with_label) # flush previous sequence
                if with_label:
                    line = line[1:].strip()
                    if split_label:
                        line = line.split()[0]
                    cur_seq_label = line
            else:   # sequence line
                buf.append(line.strip())

    _flush_current_seq(with_label)

    if with_label:
        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), 'Found duplicate sequence labels'
        return sequence_labels, sequence_strs
    else:
        return sequence_strs
