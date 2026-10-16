"""Download and unpack the HDF5 packages of the pynbody test data (https://osf.io/5m6zs/) into tests/testdata.

Uses only the standard library, so that it can run before anything is installed. Run it from anywhere:

    python tests/fetch_testdata.py           # fetch any packages not already present
    python tests/fetch_testdata.py --hash    # print a hash identifying the packages, for caching in CI
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

osf_project_id = '5m6zs'

# The packages of HDF5 files used, each with a path whose presence shows it has been unpacked, and where to unpack it
# (the same as in pynbody.test_utils, so that a pynbody tests/testdata directory can be used instead)
packages = {
    'swift.tar.gz': ('SWIFT', ''),
    'swift_isolated.tar.gz': ('SWIFT/isolated_0008.hdf5', 'SWIFT'),
    'swift_planetary.tar.gz': ('SWIFT/planetary.hdf5', 'SWIFT'),
    'gadget.tar.gz': ('gadget3', ''),
    'arepo.tar.gz': ('arepo', ''),
    'gizmo.tar.gz': ('gizmo', ''),
}

testdata = pathlib.Path(__file__).parent / 'testdata'


def packages_hash() -> str:
    m = hashlib.sha256(osf_project_id.encode())
    for name in sorted(packages):
        m.update(name.encode())
    return m.hexdigest()


def _get(url: str, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'bulkread-tests'}),
                                          timeout=60)
        except OSError as e:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"Fetching {url} failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def download_urls() -> dict[str, str]:
    """The download URL of each file in the OSF project's storage"""
    urls = {}
    url = f'https://api.osf.io/v2/nodes/{osf_project_id}/files/osfstorage/?page%5Bsize%5D=100'
    while url:
        with _get(url) as response:
            listing = json.load(response)
        for item in listing['data']:
            if item['attributes']['kind'] == 'file':
                urls[item['attributes']['name']] = item['links']['download']
        url = listing['links'].get('next')
    return urls


def fetch(verbose=True):
    wanted = [name for name, (verify, _) in packages.items() if not (testdata / verify).exists()]
    if not wanted:
        if verbose:
            print("All test data packages are present")
        return
    urls = download_urls()
    for name in wanted:
        unpack_to = testdata / packages[name][1]
        unpack_to.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Downloading {name}")
        with tempfile.TemporaryFile() as archive:
            with _get(urls[name]) as response:
                shutil.copyfileobj(response, archive)
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
