# import dask.dataframe as dd

import pandas as pd
# import torch

from pathlib import Path
from typing import Optional, Tuple, Callable, Union, Dict

# from RnaBench.download import select_and_download
from downstream_ss.RnaBench.lib.tasks import RNA
from downstream_ss.RnaBench.lib.alphabets import (
    AlphabetConverter,
    Nucleotide,
    Structure,
    get_nuc_vocab,
    get_struc_vocab,
)

from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")


class RnaDataset():
    def __init__(self,
                 dataset: Union[Path, pd.DataFrame],
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = True,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 sequence_vocab: Optional[Tuple[str]] = Nucleotide.iupac_alphabet,
                 structure_vocab: Optional[Tuple[str]] = Structure.extended_dot_bracket,
                 feature_extractors: Optional[Dict[str, Callable]] = None,
                 matrix: Optional[bool] = None,
                 ):
        
        self._nc = nc
        self._pks = pks
        self._multiplets = multiplets
        self._min_length = min_length
        self._max_length = max_length
        self._matrix = matrix

        if not isinstance(dataset, pd.DataFrame):
            self.data = RnaDataset.load(dataset=dataset,
                                        nc=self._nc,
                                        pks=self._pks,
                                        multiplets=self._multiplets,
                                        min_length=self._min_length,
                                        max_length=self._max_length,
                                        )
        else:
            self.data = dataset

        self._max_seq_length = self.data['length'].max()
        self.data.loc[:, 'pair_length'] = self.data['pairs'].apply(len)
        self.max_pair_length = self.data['pair_length'].max()

        if sequence_vocab is None:
            self.sequence_vocab = get_nuc_vocab(self.data)
        else:
            self.sequence_vocab = sequence_vocab
        if structure_vocab is None:
            self.structure_vocab = get_struc_vocab(self.data)
        else:
            self.structure_vocab = structure_vocab

        self.seq_alphabet_converter = AlphabetConverter(self.sequence_vocab)
        self.struc_alphabet_converter = AlphabetConverter(self.structure_vocab)

        if feature_extractors is not None:
            self.feature_extractors = feature_extractors
            self.distribution = None
            self.descriptors_distribution = {}
            self.data = self.extract_features(self.data, self.feature_extractors)
            # print(self.data)

    @staticmethod
    def load(
            dataset: Path,
            nc: bool,
            pks: bool,
            multiplets: bool,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            feature_extractors: Optional[Dict[str, Callable]] = None,
    ):
        if not Path(dataset).is_file():
            # Path(self.data_dir).mkdir(exist_ok=True, parents=True)
            # t = '_'.join(str(Path(dataset).stem).split('_')[:-1])
            # select_and_download(task=t,
            #                     save_dir='data',
            #                     )
            print("No Data!")

        data = pd.read_pickle(dataset)

        if min_length is not None:
            data = data[data['sequence'].apply(lambda x: min_length <= len(x))]
        if max_length is not None:
            data = data[data['sequence'].apply(lambda x: len(x) <= max_length)]

        if not nc:
            data = data[data['has_nc'] == False] if 'has_nc' in data.columns else data

        if not pks:
            data = data[data['has_pk'] == False] if 'has_pk' in data.columns else data

        if not multiplets:
            data = data[data['has_multiplet'] == False] if 'has_multiplet' in data.columns else data

        if feature_extractors is not None:
            data = RnaDataset.extract_features(data, feature_extractors)
        return data

    @staticmethod
    def to_rna(l):
        return RNA(rna_id=l[0],
                   sequence=l[1],
                   pairs=l[2],
                   gc=l[3],
                   length=l[4],
                   matrix=l[5],
                   )

    def __iter__(self):
        yield from map(RnaDataset.to_rna, (list(tup) + [self._matrix] for tup in zip(self.data['Id'],
                                                                                     self.data['sequence'],
                                                                                     self.data['pairs'],
                                                                                     self.data['gc_content'],
                                                                                     self.data['length'],
                                                                                     )))

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        d = self.data.iloc[idx, :]

        return RnaDataset.to_rna((d['Id'], d['sequence'], d['pairs'], d['gc_content'], d['length'], self._matrix))

    def __len__(self):
        return len(self.data)

    @property
    def seq_stoi(self):
        return self.seq_alphabet_converter.stoi

    @property
    def struc_stoi(self):
        return self.struc_alphabet_converter.stoi

    @property
    def seq_itos(self):
        return self.seq_alphabet_converter.itos

    @property
    def struc_itos(self):
        return self.struc_alphabet_converter.itos

    @property
    def train(self):
        return RnaDataset(self.train_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix,
                          )

    @property
    def valid(self):
        return RnaDataset(self.validation_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix
                          )

    @property
    def test(self):
        return RnaDataset(self.test_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix
                          )

    @property
    def train_data(self):
        return self.data[self.data['origin'].str.contains('train')]

    @property
    def validation_data(self):
        return self.data[self.data['origin'].str.contains('valid')]

    @property
    def test_data(self):
        return self.data[~(self.data['origin'].str.contains('valid') | self.data['origin'].str.contains('train'))]

    @property
    def has_train(self):
        return not self.train_data.empty

    @property
    def has_validation(self):
        return not self.validation_data.empty

    @property
    def has_test(self):
        return not self.test_data.empty

    @property
    def max_seq_length(self):
        return self._max_seq_length

    # def dask(self, n_workers=4):
    #     self.data = dd.from_pandas(self.data, npartitions=n_workers)
    #     return self

    # def to_pandas(self):
    #     if isinstance(self.data, dd.DataFrame):
    #         self.data = self.data.compute()
    #     return self


if __name__ == '__main__':
    # get dataset as a argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='inverse_rna_folding_benchmark')
    parser.add_argument('--data-dir', type=str, default='data')

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    data_path = Path(f'{data_dir}/{dataset}')

    # dataset = 'inverse_rna_folding_benchmark'
    data = RnaDataset.load(data_path,
                           nc=False,
                           pks=False,
                           multiplets=False,
                           )

    # save dataframe in a pickle file in directory data_preprocessed

    # make sure to create the directory data_preprocessed before running the code
    # make directory data_preprocessed in the root directory of the project
    Path('data_preprocessed', *data_path.parts[:-1]).mkdir(parents=True, exist_ok=True)
    # change only the first dir in the path to data_preprocessed
    data_save_path = Path("data_preprocessed", *data_path.parts)
    data.to_pickle(f'{str(data_save_path)}')

    # print(data.distribution.loc, data.distribution.covariance_matrix)
    # print(data.data.head())