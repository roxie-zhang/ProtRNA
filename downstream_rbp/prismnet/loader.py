import os, h5py
import os.path
import numpy as np
import torch
import torch.utils.data

class SeqicSHAPE(torch.utils.data.Dataset):
    def __init__(self, data_path, is_test=False, is_val=False, is_infer=False, use_structure=True):
        """data loader
        
        Args:
            data_path ([str]): h5 file path
            is_test (bool, optional): testset or not. Defaults to False.
            is_val (bool, optional): valset or not. Defaults to False.
        """
        if is_infer:
            self.dataset = self.__load_infer_data__(data_path, use_structure=use_structure)
            print("infer data: ", self.__len__()," use_structure: ", use_structure)
        else:
            dataset = h5py.File(data_path, 'r')
            X_train = np.array(dataset['X_train']).astype(np.float32)
            Y_train = np.array(dataset['Y_train']).astype(np.int32)
            X_test  = np.array(dataset['X_test']).astype(np.float32)
            Y_test  = np.array(dataset['Y_test']).astype(np.int32)
            if len(Y_train.shape) == 1:
                Y_train = np.expand_dims(Y_train, axis=1)
                Y_test  = np.expand_dims(Y_test, axis=1)
            X_train = np.expand_dims(X_train, axis=3).transpose([0, 3, 2, 1]) # -> N, 1, L（101）, D（one-hot 4 + icshape)
            X_test  = np.expand_dims(X_test,  axis=3).transpose([0, 3, 2, 1])

            datasize = len(X_train)
            np.random.seed(int(os.environ['PYTHONHASHSEED']))
            indices = np.random.permutation(datasize)
            val_split_index = int(np.floor(0.2 * datasize)) # default val_ratio=0.2
            val_indices, train_indices = indices[:val_split_index], indices[val_split_index:]

            X_val, Y_val = X_train[val_indices], Y_train[val_indices]
            X_train, Y_train = X_train[train_indices], Y_train[train_indices]

            train = {'inputs': X_train, 'targets': Y_train}
            val = {'inputs': X_val, 'targets': Y_val}
            test  = {'inputs': X_test,  'targets': Y_test}

            labels, nums = np.unique(Y_train, return_counts=True)
            print("train:", labels, nums)
            labels, nums = np.unique(Y_val, return_counts=True)
            print("val:", labels, nums)
            labels, nums = np.unique(Y_test, return_counts=True)
            print("test:", labels, nums)

            # train = self.__prepare_data__(train)
            # test  = self.__prepare_data__(test)

            print('train shape:', train['inputs'].shape)
            print('val shape:', val['inputs'].shape)
            print('test shape:', test['inputs'].shape)

            if is_test:
                self.dataset = test
            elif is_val:
                self.dataset = val
            else:
                self.dataset = train

        

    def __load_infer_data__(self, data_path, use_structure=True):
        from prismnet.utils import datautils
        dataset = datautils.load_testset_txt(data_path, use_structure=use_structure, seq_length=101)
        return dataset
       
    
    def __prepare_data__(self, data):
        inputs    = data['inputs'][:,:,:,:4] # N, 1, L, 4
        structure = data['inputs'][:,:,:,4:] # N, 1, L ,F or 1
        structure = np.expand_dims(structure[:,:,:,0], axis=3) # N, 1, L, F or 1
        inputs    = np.concatenate([inputs, structure], axis=3)
        data['inputs']  = inputs # N, 1, L, D (5)
        return data

    def __to_sequence__(self, x):
        x1 = np.zeros_like(x[0,:,:1])
        for i in range(x1.shape[0]):
            # import pdb; pdb.set_trace()
            x1[i] = np.argmax(x[0,i,:4])
            # import pdb; pdb.set_trace()
        return x1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.dataset['inputs'][index]
        # x = self.__to_sequence__(x)
        y = self.dataset['targets'][index]
        return x, y


    def __len__(self):
        return len(self.dataset['inputs'])

