"""Reading large amounts of data from files on threads.

Bulk readers plan their reads as jobs, each an input step and a computation on what it read, and run them through
one pipeline of input and decode threads (:mod:`.pipeline`), with as many threads of each kind as suits the
filesystem and machine (:mod:`.filesystem`). The reader of HDF5 datasets (:mod:`bulkread.hdf`) is built this way,
and readers of other formats can share the same machinery.
"""

from . import filesystem, pipeline
from .pipeline import Job, ReadStrategy, perform_tasks

__version__ = "0.1.0"

__all__ = ['Job', 'ReadStrategy', 'filesystem', 'perform_tasks', 'pipeline']
