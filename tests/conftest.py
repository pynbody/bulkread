import numpy as np
import pytest


class _StandInRemoteDataset:
    """Stands in for a remote dataset (such as hdfstream's RemoteDataset), by wrapping an h5py dataset in an object
    that is not one, and that reads as hdfstream's does: read_direct only into C-contiguous arrays, converting only
    where that is safe; and indexing by an array of rows, recording every such array it is given"""

    def __init__(self, dataset):
        self._dataset = dataset
        self.shape, self.dtype, self.ndim, self.attrs = dataset.shape, dataset.dtype, dataset.ndim, dataset.attrs
        self.index_arrays = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        if isinstance(rows, (np.ndarray, list)):
            self.index_arrays.append(np.array(rows))
            return self._dataset[np.asarray(rows)]
        return self._dataset[key]

    def read_direct(self, array, source_sel=None, dest_sel=None):
        if not array.flags.c_contiguous:
            raise RuntimeError("Destination for read_direct() must be C contiguous")
        if array.dtype != self.dtype and not np.can_cast(self.dtype, array.dtype, casting='safe'):
            raise RuntimeError(f"Cannot safely cast {self.dtype} to {array.dtype}")
        self._dataset.read_direct(array, source_sel=source_sel)


@pytest.fixture
def stand_in_remote_dataset():
    """The class _StandInRemoteDataset, for tests of reading datasets other than h5py's"""
    return _StandInRemoteDataset
