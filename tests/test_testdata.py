"""Reading the HDF5 files of the pynbody test data (real snapshots written by SWIFT, GADGET, AREPO and GIZMO) through
the bulk reader, and checking that what it reads is exactly what h5py reads.

The data are fetched by tests/fetch_testdata.py into tests/testdata (or are looked for in the directory named by the
BULKREAD_TESTDATA environment variable). Without them, these tests are skipped, unless BULKREAD_REQUIRE_TESTDATA is
set, as in CI, when they fail.
"""

import os
import pathlib
import warnings

import h5py
import numpy as np
import pytest

from bulkread import hdf

testdata = pathlib.Path(os.environ.get('BULKREAD_TESTDATA', pathlib.Path(__file__).parent / 'testdata'))

_HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'


def _is_hdf5(path: pathlib.Path) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(8) == _HDF5_SIGNATURE
    except OSError:
        return False


def _hdf5_files():
    if not testdata.is_dir():
        return []
    return sorted(p for p in testdata.rglob('*') if p.is_file() and not p.is_symlink() and _is_hdf5(p))


hdf5_files = _hdf5_files()

if not hdf5_files:
    if os.environ.get('BULKREAD_REQUIRE_TESTDATA'):
        raise RuntimeError(f"No HDF5 test data found in {testdata}; run tests/fetch_testdata.py")
    pytest.skip(f"no HDF5 test data in {testdata} (run tests/fetch_testdata.py)", allow_module_level=True)

pytestmark = pytest.mark.testdata


def _file_id(path):
    return str(path.relative_to(testdata))


def _datasets(f: h5py.File) -> list[h5py.Dataset]:
    """The datasets of numbers in a file, with at least one row"""
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 and obj.shape[0] > 0 and obj.dtype.kind in 'iuf':
            found.append(obj)

    f.visititems(visit)
    return found


@pytest.fixture
def reader():
    r = hdf.BulkReader()
    yield r
    r.close()


@pytest.fixture(autouse=True)
def read_time_fallbacks_fail():
    """A fallback to h5py while reading (rather than on opening a dataset, where it is decided from the metadata) means
    the bulk reader found a file to hold something other than it expected: in these files, that is a failure"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', hdf.BulkReadFallbackWarning)
        yield
    read_time = [str(w.message) for w in caught
                 if issubclass(w.category, hdf.BulkReadFallbackWarning) and "could not read" in str(w.message)]
    assert not read_time, read_time


def _row_selections(n):
    return [slice(0, n), slice(0, 1), slice(n - 1, n), slice(n // 3, 2 * n // 3 + 1), slice(n // 2, n // 2)]


@pytest.mark.parametrize("filename", hdf5_files, ids=_file_id)
def test_every_dataset_matches_h5py(filename, reader):
    with h5py.File(filename, 'r') as f:
        for dataset in _datasets(f):
            expected = dataset[()]
            n = dataset.shape[0]
            for rows in _row_selections(n):
                got = np.empty_like(expected[rows])
                reader.read([hdf.ReadRequest(dataset, rows, got)])
                np.testing.assert_array_equal(got, expected[rows], err_msg=f"{dataset.name}[{rows}]")
            rows = np.unique(np.random.default_rng(1).integers(0, n, size=max(n // 7, 1)))
            got = np.empty((len(rows),) + expected.shape[1:], dtype=expected.dtype)
            reader.read([hdf.ReadRequest(dataset, rows, got)])
            np.testing.assert_array_equal(got, expected[rows], err_msg=f"{dataset.name}[index array]")


_conversion_nbytes = 64 * 1024 ** 2


@pytest.mark.parametrize("filename", hdf5_files, ids=_file_id)
def test_conversion_to_wider_types_matches_h5py(filename, reader):
    with h5py.File(filename, 'r') as f:
        for dataset in _datasets(f):
            wider = np.dtype(np.float64 if dataset.dtype.kind == 'f' else np.int64)
            if dataset.dtype.kind == 'u' and dataset.dtype.itemsize == 8:
                continue
            # (at most _conversion_nbytes of the converted rows, which is several of the blocks they are converted in)
            row_nbytes = wider.itemsize * int(np.prod(dataset.shape[1:]))
            rows = slice(0, min(dataset.shape[0], max(_conversion_nbytes // row_nbytes, 1)))
            expected = np.empty((rows.stop,) + dataset.shape[1:], dtype=wider)
            dataset.read_direct(expected, source_sel=np.s_[rows])
            got = np.empty_like(expected)
            reader.read([hdf.ReadRequest(dataset, rows, got)])
            np.testing.assert_array_equal(got, expected, err_msg=dataset.name)


# The most data (in bytes, decoded) read in one call by test_all_files_read_together_on_threads, so that the test
# fits in the memory of a CI runner
_batch_nbytes = 500 * 1024 ** 2


def _batches_of_files():
    """The files, in batches of consecutive files (so that, for example, the files of a multi-file snapshot are mostly
    read together), each holding at most _batch_nbytes of data unless a single file holds more"""
    batches, batch, total = [], [], 0
    for filename in hdf5_files:
        with h5py.File(filename, 'r') as f:
            nbytes = sum(d.size * d.dtype.itemsize for d in _datasets(f))
        if batch and total + nbytes > _batch_nbytes:
            batches.append(batch)
            batch, total = [], 0
        batch.append(filename)
        total += nbytes
    return batches + [batch]


@pytest.mark.parametrize("threads", [(1, 0), (2, 0), (1, 3), (4, 4)])
def test_all_files_read_together_on_threads(monkeypatch, reader, threads):
    """Every dataset of every file, read a batch of files in one call, with input and decode threads (where there is
    work for them)"""
    monkeypatch.setitem(hdf.strategy.config, 'io-threads', threads[0])
    monkeypatch.setitem(hdf.strategy.config, 'decode-threads', threads[1])
    for batch in _batches_of_files():
        files = [h5py.File(filename, 'r') for filename in batch]
        try:
            datasets = [d for f in files for d in _datasets(f)]
            destinations = [np.empty(d.shape, dtype=d.dtype) for d in datasets]
            reader.read([hdf.ReadRequest(d, slice(None), a) for d, a in zip(datasets, destinations)])
            for dataset, got in zip(datasets, destinations):
                np.testing.assert_array_equal(got, dataset[()], err_msg=f"{dataset.file.filename}:{dataset.name}")
            del destinations, got
        finally:
            for f in files:
                f.close()


# Snapshots whose particle data are all stored in ways the bulk reader reads directly (as pynbody's tests also check)
directly_read = ['SWIFT/snap_0150.hdf5', 'SWIFT/multifile_with_vds/snap_0000.hdf5', 'SWIFT/planetary.hdf5',
                 'gadget3/data/snapshot_103/snap_103.hdf5', 'gadget3/snap_028_z000p000.0.hdf5', 'arepo/agora_100.hdf5']


@pytest.mark.parametrize("name", directly_read)
def test_particle_data_are_read_directly(name, reader):
    """The other tests would pass trivially if everything fell back to h5py, so check that the particle data of
    snapshots are read directly"""
    if not (testdata / name).exists():
        if os.environ.get('BULKREAD_REQUIRE_TESTDATA'):
            pytest.fail(f"{name} is not in the test data")
        pytest.skip(f"{name} is not in the test data")
    with h5py.File(testdata / name, 'r') as f:
        particle_data = [d for d in _datasets(f) if d.parent.name.startswith('/PartType')]
        assert particle_data
        through_h5py = [d.name for d in particle_data if not hdf.is_direct_reader(reader.open(d))]
        assert not through_h5py, f"read through h5py: {through_h5py}"
