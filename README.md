# bulkread

Fast, threaded bulk reads of large datasets, in particular of HDF5 simulation snapshots.

h5py serialises every call into libhdf5 behind one interpreter-wide lock, so reads of different files, and the
decompression of different chunks, cannot overlap. `bulkread.hdf` uses h5py only for the cheap part (finding a
dataset, its datatype, layout, filters and where its bytes live) and then reads the bytes itself, undoing the
deflate, shuffle and fletcher32 filters in numpy and zlib (which releases the GIL). Reads are spread over input threads
(several files at once on a parallel filesystem such as Lustre; one reader on a local disk) and decode threads (for
compressed data).

Nothing is read behind HDF5's back unless bulkread can be sure of reading it exactly as HDF5 would: any dataset
stored in a way it does not handle (other filters, drivers, datatypes, layouts) is read through h5py, with a
`BulkReadFallbackWarning`. Virtual datasets, such as the single-file view SWIFT writes of a multi-file snapshot, are
decomposed into their sources, which are read directly.

The package was extracted from [pynbody](https://github.com/pynbody/pynbody), so that other readers can share it.

## Installation

```
pip install .
```

The only dependencies are numpy and h5py.

## Usage

```python
import h5py
import numpy as np
from bulkread import hdf

reader = hdf.BulkReader()  # keep one for a set of files that are read repeatedly

with h5py.File("snap_0000.hdf5", "r") as f:
    coordinates = f["PartType0/Coordinates"]
    masses = f["PartType0/Masses"]
    pos = np.empty(coordinates.shape, dtype=coordinates.dtype)
    m = np.empty(1000, dtype=np.float64)  # (converted as it is read)
    strategy = reader.read([hdf.ReadRequest(coordinates, slice(None), pos),
                            hdf.ReadRequest(masses, slice(0, 1000), m)])
    print(strategy.reason)

reader.close()
```

Rows may also be given as a sorted array of row numbers, in which case only the chunks holding them are decoded.

The number of threads is chosen automatically (see `bulkread.hdf.strategy`), and can be overridden:

```python
hdf.strategy.config["decode-threads"] = 4
hdf.strategy.configure({"io-threads": "2"}, source="my configuration file")  # (checks the values)
```

The pipeline of input and decode threads (`bulkread.pipeline`) and the knowledge of filesystems
(`bulkread.filesystem`) are independent of HDF5, and can be used by readers of other formats: pynbody's RAMSES
reader uses them.

## Tests

```
pip install . pytest
python tests/fetch_testdata.py   # the HDF5 packages of the pynbody test data (about 850 MB), from https://pynbody.pontzen.uk
python -m pytest
```

Most tests write their own small HDF5 files; `tests/test_testdata.py` checks that every dataset of the real snapshots
in the pynbody test data reads exactly as it does through h5py, and is skipped if the data have not been fetched.

## Licence

bulkread is free software, under the GNU Lesser General Public License, version 3 or (at your option) any later
version: see `COPYING.LESSER` and `COPYING`.
