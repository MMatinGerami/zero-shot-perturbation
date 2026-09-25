"""Stream a large single-cell prediction to an .h5ad without holding it in memory.

A challenge submission is ~360k cells x 18.5k genes with ~6k stored counts per cell:
about 2 billion nonzeros, more than fits comfortably in RAM on a laptop when assembled
with `scipy.sparse.vstack`. This writer appends CSR blocks straight into resizable HDF5
datasets in the layout `anndata` reads back as `X`, then writes `obs`/`var` at the end.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

_CHUNK = 1 << 20


class StreamingH5ad:
    """Append CSR blocks row-wise into an .h5ad; `var` fixed up-front, `obs` collected."""

    def __init__(self, path: str | Path, var: pd.DataFrame, n_obs: int):
        self.path = Path(path)
        self.var = var
        self.n_obs = n_obs
        self._obs: list[pd.DataFrame] = []
        self._rows = 0
        self._nnz: int | None = None
        self._f = h5py.File(self.path, "w")
        self._f.attrs["encoding-type"] = "anndata"
        self._f.attrs["encoding-version"] = "0.1.0"
        g = self._f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array([n_obs, len(var)], dtype=np.int64)
        self._data = g.create_dataset(
            "data", (0,), maxshape=(None,), dtype=np.float32, chunks=(_CHUNK,)
        )
        self._indices = g.create_dataset(
            "indices", (0,), maxshape=(None,), dtype=np.int32, chunks=(_CHUNK,)
        )
        self._indptr = g.create_dataset("indptr", (n_obs + 1,), dtype=np.int64)
        self._indptr[0] = 0

    @property
    def nnz(self) -> int:
        if self._nnz is not None:
            return self._nnz
        return int(self._data.shape[0])

    def append(self, block: sp.csr_matrix, obs: pd.DataFrame) -> None:
        block = sp.csr_matrix(block)
        block.eliminate_zeros()
        if block.shape[0] != len(obs) or block.shape[1] != len(self.var):
            raise ValueError(
                f"block {block.shape} does not match obs {len(obs)} / var {len(self.var)}"
            )
        start = self.nnz
        n = block.nnz
        self._data.resize((start + n,))
        self._indices.resize((start + n,))
        self._data[start:] = block.data.astype(np.float32, copy=False)
        self._indices[start:] = block.indices.astype(np.int32, copy=False)
        r0 = self._rows
        self._indptr[r0 + 1 : r0 + block.shape[0] + 1] = start + block.indptr[1:].astype(np.int64)
        self._rows += block.shape[0]
        self._obs.append(obs)

    def close(self) -> pd.DataFrame:
        if self._rows != self.n_obs:
            raise ValueError(f"wrote {self._rows} rows, declared {self.n_obs}")
        obs = pd.concat(self._obs, ignore_index=True)
        obs.index = pd.Index([f"cell{i}" for i in range(self.n_obs)], name="obs_names")
        for col in obs.columns:
            if obs[col].dtype == object:
                obs[col] = obs[col].astype("category")
        self._nnz = self.nnz
        ad.io.write_elem(self._f, "obs", obs)
        ad.io.write_elem(self._f, "var", self.var)
        for name in ("obsm", "varm", "obsp", "varp", "uns", "layers"):
            grp = self._f.create_group(name)
            grp.attrs["encoding-type"] = "dict"
            grp.attrs["encoding-version"] = "0.1.0"
        self._f.close()
        return obs
