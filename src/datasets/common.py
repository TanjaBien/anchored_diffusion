import numpy as np
import pandas as pd
import torch

from src.utils.tensors import generalized_b_xy_c_to_image


def cycle(dl):
    while True:
        for data_item in dl:
            yield data_item


class SingleTensorDataset:
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


class HDF5TensorDataset:
    def __init__(self, path, key='tensor', dtype=torch.float32):
        self.path = str(path)
        self.key = key
        self.dtype = dtype
        self._file = None
        self._dataset = None

        import h5py

        with h5py.File(self.path, 'r') as f:
            if self.key not in f:
                raise KeyError(f"HDF5 file '{self.path}' does not contain dataset '{self.key}'.")
            self.raw_shape = tuple(f[self.key].shape)

        self.shape = self._normalized_shape(self.raw_shape)
        self.num_datapoints = self.shape[0]

    @staticmethod
    def _normalized_shape(raw_shape):
        if len(raw_shape) == 3:
            return raw_shape[0], 1, raw_shape[1], raw_shape[2]
        if len(raw_shape) == 4 and raw_shape[-1] == 1 and raw_shape[1] != 1:
            return raw_shape[0], raw_shape[-1], raw_shape[1], raw_shape[2]
        return raw_shape

    def _get_dataset(self):
        if self._dataset is None:
            import h5py

            self._file = h5py.File(self.path, 'r')
            self._dataset = self._file[self.key]
        return self._dataset

    def close(self):
        file_handle = self._file
        self._file = None
        self._dataset = None
        if file_handle is not None:
            file_handle.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        state['_dataset'] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        sample = self._get_dataset()[idx]
        sample = torch.as_tensor(sample, dtype=self.dtype)
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3 and sample.shape[-1] == 1 and sample.shape[0] != 1:
            sample = sample.permute(2, 0, 1)
        return sample


class DarcyDataset:
    """Pressure and permeability fields from <split_dir>/{p_data,K_data}.csv, one field per row,
    as (n, 2, pixels, pixels): channel 0 = p, channel 1 = K."""

    def __init__(self, split_dir):
        fields = [pd.read_csv(f'{split_dir}/{name}.csv', header=None) for name in ('p_data', 'K_data')]
        data = torch.tensor(np.stack(fields, axis=-1), dtype=torch.float32)
        self.data = generalized_b_xy_c_to_image(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
