"""Download and unpack the HDF5 packages of the pynbody test data into tests/testdata.

The archives come from the pynbody data store (https://pynbody.pontzen.uk/testdata/), or from a mirror named by the
PYNBODY_DATA_URL environment variable. Uses only the standard library, so that it can run before anything is
installed. Run it from anywhere:

    python tests/fetch_testdata.py           # fetch any packages not already present
    python tests/fetch_testdata.py --hash    # print a hash identifying the packages, for caching in CI
"""

import argparse
import hashlib
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

data_url = os.environ.get('PYNBODY_DATA_URL', 'https://pynbody.pontzen.uk').rstrip('/')

# The packages of HDF5 files used, each with a path whose presence shows it has been unpacked, and where to unpack it
# (as in pynbody.test_utils, so that a pynbody tests/testdata directory can be used instead)
packages = {
    'swift.tar.gz': ('SWIFT/snap_0150.hdf5', ''),
    'swift_isolated.tar.gz': ('SWIFT/isolated_0008.hdf5', 'SWIFT'),
    'swift_planetary.tar.gz': ('SWIFT/planetary.hdf5', 'SWIFT'),
    'gadget.tar.gz': ('gadget3', ''),
    'arepo.tar.gz': ('arepo/cosmobox_015.hdf5', ''),
    'gizmo.tar.gz': ('gizmo', ''),
}

testdata = pathlib.Path(__file__).parent / 'testdata'


def packages_hash() -> str:
    m = hashlib.sha256(data_url.encode())
    for name in sorted(packages):
        m.update(name.encode())
    return m.hexdigest()


def _download(url: str, destination, attempts: int = 5):
    for attempt in range(attempts):
        try:
            destination.seek(0)
            destination.truncate()
            request = urllib.request.Request(url, headers={'User-Agent': 'bulkread-tests'})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = response.headers.get('Content-Length')
                shutil.copyfileobj(response, destination)
            # (a connection cut short can look like the end of the file, so check, before anything is unpacked)
            if expected is not None and destination.tell() != int(expected):
                raise OSError(f"received {destination.tell()} of {expected} bytes")
            return
        except OSError as e:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"Fetching {url} failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def fetch(verbose=True):
    for name, (verify, unpack_path) in packages.items():
        if (testdata / verify).exists():
            if verbose:
                print(f"{name} is already present")
            continue
        unpack_to = testdata / unpack_path
        unpack_to.mkdir(parents=True, exist_ok=True)
        url = f"{data_url}/testdata/{name}"
        if verbose:
            print(f"Downloading {url}")
        with tempfile.TemporaryFile() as archive:
            _download(url, archive)
            archive.seek(0)
            if verbose:
                print(f"Unpacking {name} into {unpack_to}")
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(unpack_to, filter='data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hash', action='store_true', help="print a hash identifying the packages, and exit")
    args = parser.parse_args()
    if args.hash:
        print(packages_hash())
    else:
        fetch()
