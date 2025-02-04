import pickle
import pandas as pd
import numpy as np

from enum import Enum, unique
from typing import List, Optional
from pathlib import Path
from datetime import datetime

# from downstream_ss.RnaBench.download import select_and_download
from downstream_ss.RnaBench.lib.datasets import RnaDataset
# from downstream_ss.RnaBench.lib.execution import timing
from downstream_ss.RnaBench.lib.metrics import GoalDirectedMetrics
from downstream_ss.RnaBench.lib.alphabets import Nucleotide, Structure


@unique
class Tasks(Enum):
    intraFamily = 'intra_family'
    interFamily = 'inter_family'
    inter_family_fine_tuning = "inter_family_fine_tuning"
    intra_family_valid = "intra_family_valid"
    inter_family_valid = "inter_family_valid"
    bprna_val = "bprna_val"
    bprna_test = "bprna_test"


class Benchmark():
    def __init__(self, **kwargs):
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    def load_data(self, save_new_features=True):
        if self.task.value == 'inter_family_fine_tuning':
            data_path = Path(self.data_dir, 'inter_family_benchmark.plk')
        elif "valid" in self.task.value or "archive" in self.task.value:
            data_path = Path(self.data_dir, f'{self.task.value}.plk')
        elif "bprna" in self.task.value:
            if "val" in self.task.value:
                data_path = Path(self.data_dir, 'bprna_VL0.plk')
            elif "test" in self.task.value:
                data_path = Path(self.data_dir, 'bprna_TS0.plk')
        else:
            data_path = Path(self.data_dir, self.task.value + '_benchmark.plk')

        if not data_path.is_file():
            print("No data!")
            # Path(self.data_dir).mkdir(exist_ok=True, parents=True)
            # select_and_download(self.task.value,
            #                     save_dir=self.data_dir,
            #                     )

        self.data = pd.read_pickle(data_path)
        print("loaded data:", data_path)
        print("dataset length:", len(self.data))
        if self.min_length is not None:
            self.data = self.data[self.data['sequence'].apply(lambda x: self.min_length <= len(x))]
        if self.max_length is not None:
            self.data = self.data[self.data['sequence'].apply(lambda x: len(x) <= self.max_length)]

        if not self.nc:
            self.data = self.data[self.data['has_nc'] == False]

        if not self.pks:
            self.data = self.data[self.data['has_pk'] == False]

        if not self.multiplets:
            self.data = self.data[self.data['has_multiplet'] == False]


    def get_datasets(self,
                     matrix=False,
                     sequence_vocab=Nucleotide.iupac_alphabet,
                     structure_vocab=Structure.extended_dot_bracket,
                     task=None,
                     nc=None,
                     pks=None,
                     multiplets=None,
                     min_length=None,
                     max_length=None,
                    ):
        if task is not None:
            try:
                # task = Tasks(task.lower())
                task = Tasks(task)
            except ValueError as e:
                valid_tasks = [x.value for x in Tasks]
                raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                     f"Please use one of {valid_tasks}.") from e
        else:
            task = self.task

        if nc is None:
            nc = self.nc
        if pks is None:
            pks = self.pks
        if multiplets is None:
            multiplets = self.multiplets
        if min_length is None:
            min_length = self.min_length
        if max_length is None:
            max_length = self.max_length


        if task.value == 'inter_family_fine_tuning':
            train_path = Path(self.data_dir, task.value + '_train.plk')
            valid_path = Path(self.data_dir, 'inter_family_valid.plk')
            test_path = Path(self.data_dir, 'inter_family_benchmark.plk')
        elif task.value == 'riboswitch_design':
            train_path = Path(self.data_dir, task.value + '_train.plk.gz')
            valid_path = None
            test_path = None
        else:
            train_path = Path(self.data_dir, task.value + '_train.plk')
            valid_path = Path(self.data_dir, task.value + '_valid.plk')
            test_path = Path(self.data_dir, task.value + '_benchmark.plk')

        train = RnaDataset(
          dataset=train_path,
          nc=nc,
          pks=pks,
          multiplets=multiplets,
          min_length=min_length,
          max_length=max_length,
          sequence_vocab=sequence_vocab,
          structure_vocab=structure_vocab,
          matrix=matrix,
        )
        if valid_path is not None:
            valid = RnaDataset(
              dataset=valid_path,
              nc=nc,
              pks=pks,
              multiplets=multiplets,
              min_length=min_length,
              max_length=max_length,
              sequence_vocab=sequence_vocab,
              structure_vocab=structure_vocab,
              matrix=matrix,
            )
        else:
            valid = None
        if test_path is not None:
            test = RnaDataset(
              dataset=test_path,
              nc=nc,
              pks=pks,
              multiplets=multiplets,
              min_length=min_length,
              max_length=max_length,
              sequence_vocab=sequence_vocab,
              structure_vocab=structure_vocab,
              matrix=matrix,
            )
        else:
            test = None

        return train, valid, test


class RnaFoldingBenchmark(Benchmark):

    _gd_metrics: List[str] = [
                 'f1_score',
                 'mcc',
                 # 'wl',
                 'recall',
                 'precision',
                 'specificity',
                 'solved',
                 'f1_shifted',
                 ]

    _evaluation_counter = 0

    def __init__(
                 self,
                 task : str = 'inter_family',
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = True,
                 data_dir: str = 'downstream_ss/data',
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 results_dir: str = 'results',
                ):

        # get task
        try:
            # self.task = Tasks(task.lower())
            self.task = Tasks(task)
        except ValueError as e:
            valid_tasks = [x.value for x in Tasks]
            raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                 f"Please use one of {valid_tasks}.") from e

        self.gd_metrics = GoalDirectedMetrics(self._gd_metrics)

        super().__init__(
                         min_length=min_length,
                         max_length=max_length,
                         nc=nc,
                         pks=pks,
                         multiplets=multiplets,
                         results_dir=results_dir,
                         data_dir=data_dir,
                        )
        self.load_data()


    def __call__(self,
                 wrapper_function,
                 *args,
                 save_results=True,
                 dataset=None,
                 results_path=None,
                 task_ids=None,
                 algorithm_name='custom',
                 dist_metrics = False,
                 **kwargs,
                 ):
        self._evaluation_counter += 1
        if dataset is not None:
            if isinstance(dataset, RnaDataset):
                self._results = dataset.data.copy()
            elif isinstance(dataset, pd.DataFrame):
                self._results = dataset.copy()
            # elif isinstance(dataset, TorchDataset):
            #     self._results = dataset.dataset.data.copy()
            else:
                raise UserWarning(f"Dataset type {type(dataset)} currently not supported")

        else:
            self._results = self.data.copy()

        if task_ids is not None:
            if isinstance(task_ids, int):
                self._results = self._results[self._results['Id'].isin([task_ids])]
            elif isinstance(task_ids, list):
                self._results = self._results[self._results['Id'].isin(task_ids)]
            elif isinstance(task_ids, np.ndarray):
                self._results = self._results[self._results['Id'].isin([task_ids.tolist()])]
            else:
                raise UserWarning(f"Task_ids should be of type <int> or <list> or <numpy.ndarray>, found: {type(task_ids)}")

        self._results.loc[:, 'predicted_sequence'] = self._results['sequence']
        if "val" not in self.task.value:
            print(f"generating pairs from algorithm {algorithm_name}...")
        

        # self._results[['predicted_pairs', 'time']] = self._results.apply(lambda x: self._predict(x, wrapper_function, *args, **kwargs), axis=1, result_type='expand')
        self._results['predicted_pairs'] = self._results.apply(lambda x: self._predict(x, wrapper_function, *args, **kwargs), axis=1)
   
        self._results = self._results[self._results['pairs'].apply(lambda x: x != [])]
        predicted_columns = [col for col in self._results.columns if col.startswith('predicted_') or col == 'Id']
        pred_df = self._results.filter(predicted_columns)
        if "val" not in self.task.value:
            print("prediction finished, applying metrics...")
        # Create a dictionary to map old column names to new column names
        column_mapping = {col: col.replace('predicted_', '') for col in predicted_columns}
        # Create a new dataframe with renamed columns
        pred_df = pred_df.rename(columns=column_mapping)
        #get column names before
        old_cols = pred_df.columns

        #drop old columns
        pred_df = pred_df.drop(columns=old_cols)
        #rename columns with predicted_ prefix
        pred_df.columns = ['predicted_' + col for col in pred_df.columns]
        #add pred_df columns to self._results if not already present
        self._results = pd.concat([self._results, pred_df], axis=1)
        self._results.loc[:, 'per_sample_metrics'] = self._results.apply(lambda x: self.gd_metrics(x), axis=1)

        per_sample_metrics = pd.DataFrame.from_dict(self._results['per_sample_metrics'].to_list())
        per_sample_metrics.index = self._results.index

        # requires workaround: unsure where wl comes from from tiem to time... TODO: find out and debug again!
        if 'wl' in per_sample_metrics.columns:
            per_sample_metrics.drop('wl', axis=1, inplace=True)

        self._results = pd.concat([self._results, per_sample_metrics], axis=1)
        if dist_metrics:
            self.df_results = self.dl_metrics.evaluate(self._results)
        
        per_sample_metrics = per_sample_metrics.fillna(0)

        scores = {m: np.mean(per_sample_metrics[m])
                         if not 'solved' in m or not 'valid' in m else np.sum(per_sample_metrics[m])
                         for m in per_sample_metrics.columns}
        if dist_metrics:
            # append df_results to scores
            scores.update({m: self.df_results[m] for m in self.df_results})

        if save_results:
            print("saving results...")
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            Path(self.results_dir, "RNA_folding", self.task.value, algorithm_name).mkdir(exist_ok=True, parents=True)

            if results_path is not None:
                out_path = Path(results_path)
            else:
                out_path = Path(self.results_dir, "RNA_folding", self.task.value, algorithm_name, f"{'_'.join([str(x) for x in [algorithm_name, self._evaluation_counter, 'min_len', self.min_length, 'max_len', self.max_length, 'nc', self.nc, 'pks', self.pks, 'multiplets', self.multiplets]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
            with open(out_path, 'wb') as f:
                pickle.dump(self._results, f)

        return {m: np.round(v, 3) for m, v in scores.items()}

    # @timing
    def _predict(self, x, wrapper_function, *args, **kwargs):
        return wrapper_function(x, *args, **kwargs)
