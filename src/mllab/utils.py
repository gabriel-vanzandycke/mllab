from torch.utils.data import Dataset


import errno
import os
from typing import Optional


class TorchDataset(Dataset):
    def __init__(self, parent):
        self.parent = parent
        self.keys = list(parent.keys)
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, i):
        return self.parent.query_item(self.keys[i])


class LazyList:
    def __init__(self, generator):
        self._generator = generator  # The generator function
        self._cache = []  # Cache to store already generated items

    def __getitem__(self, index):
        while len(self._cache) <= index:
            try:
                if self._generator is None:
                    raise StopIteration
                self._cache.append(next(self._generator))
            except StopIteration:
                self._generator = None
                raise IndexError("generator exhausted")
        return self._cache[index]

    def __iter__(self):
        yield from self._cache
        if self._generator is not None:
            for item in self._generator:
                self._cache.append(item)
                yield item
            self._generator = None

    def __len__(self):
        if self._generator is not None:
            raise ValueError("unexhausted generator as no len()")
        return len(self._cache)


class GeneratorBackedCache:
    """A cache based on a generator. When a key requested is not in cache yet,
    the generator is consumed until the key is found. Items are kept in cache
    except if `pop` is used.
    """
    def __init__(self, gen):
        self.gen = gen
        self._keys = []
        self._mapping = {}

    def _consume_next(self):
        consumed = next(self.gen, None)
        if consumed is None:
            return None
        key, value = consumed
        self._keys.append(key)
        self._mapping[key] = value
        return key, value

    def keys(self):
        i = 0
        while True:
            if i >= len(self._keys):
                ret = self._consume_next()
                if ret is None:
                    break
            yield self._keys[i]
            i += 1

    def __getitem__(self, key):
        while key not in self._keys:
            ret = self._consume_next()
            if ret is None:
                raise KeyError(key)
        try:
            return self._mapping[key]
        except KeyError as e:
            raise KeyError(f"{key} has been removed from cache.") from e

    def pop(self, key):
        item = self.__getitem__(key)
        self._mapping.pop(key)
        return item




def find(path, dirs=None, verbose=True) -> str:
    if os.path.isabs(path):
        if not os.path.isfile(path) and not os.path.isdir(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return path

    dirs = dirs or [os.getcwd(), *os.getenv("DATA_PATH", "").split(":")]
    for dirname in dirs:
        if dirname is None:
            continue
        tmp_path = os.path.join(dirname, path)
        if os.path.isfile(tmp_path) or os.path.isdir(tmp_path):
            not verbose or print("{} found in {}".format(path, tmp_path))
            return tmp_path

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                "{} (searched in {})".format(path, dirs))
