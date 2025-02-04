from dataclasses import dataclass
from typing import Tuple, Dict

# TODO: Probably have to implement classes again to have translation methods.
# And to resolve iupac nucleotides like R, Y, etc
# Can also be done in utils.utils?

@dataclass
class SpecialSymbols:
    unknown: str = '<unk>'
    bos: str = '<bos>'
    eos: str = '<eos>'
    pad: str = '<pad>'
    unknown_idx: int = 0
    pad_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3
    vocab: Tuple = (('<unk>', 0), ('<pad>', 1), ('<bos>', 2), ('<eos>', 3))
    symbols: Tuple = ('<unk>', '<pad>', '<bos>', '<eos>')

@dataclass
class Nucleotide:
    rna: tuple = ('A', 'C', 'G', 'U')
    iupac_alphabet: tuple = (
        "A",
        "C",
        "G",
        "U",
        "R",
        "Y",
        "S",
        "W",
        "K",
        "M",
        "B",
        "D",
        "H",
        "V",
        "N",
        ".",
        "-",
    )
    rna_benchmark: tuple = ('N', 'C', 'A', 'G', 'U')
    # TODO: implement modified nucleotides


@dataclass
class Structure:
    dot_bracket: tuple = (
        '.',
        '(',
        ')',
        '[',
        ']',
        '{',
        '}',
        '<',
        '>',
    )

    extended_dot_bracket: tuple = dot_bracket + tuple(chr(i) for i in range(65, 91)) + tuple(
        chr(i) for i in range(97, 123))



class AlphabetConverter():
    """
    Provides dictionaries to convert between string and numerical
    representations of alphabets.
    """
    def __init__(self, alphabet: Tuple[str]):
        self._stoi: Dict[str, int] = {i: j for (i, j) in SpecialSymbols.vocab}
        # print(self._stoi)
        self._stoi.update({s: i for s, i in zip(alphabet, range(max(self._stoi.values()) + 1,len(alphabet) + len(self._stoi.values())))})
        # print(self._stoi)
        self._itos: Dict[int, str] = {i: s for s, i in self._stoi.items()}

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos


def get_nuc_vocab(df):
    return tuple(set(df['sequence'].apply(set).apply(lambda x: ''.join(x)).sum()))

def get_struc_vocab(df, structure_col='structure'):
    return tuple(set(df['structure'].apply(set).apply(lambda x: ''.join(x)).sum()))

