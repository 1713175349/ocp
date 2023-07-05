"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform



class AseDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, path,rcut=6.0):
        from ase.io.trajectory import Trajectory
        from ocpmodels.preprocessing import AtomsToGraphs
        super(AseDataset, self).__init__()
        
        self.a2g = AtomsToGraphs(
        max_neigh=50,
        radius=rcut,
        r_energy=True,
        r_forces=True,
        r_fixed=True,
        r_distances=False,
        r_edges=True,
        )

        self.path = Path(path)
        self.cached={}
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.traj"))
            assert len(db_paths) > 0, f"No trajs found in '{self.path}'"


            self._keys, self.envs = [], []
            
            for db_path in db_paths:
                self.envs.append(Trajectory(db_path))
                length=len(self.envs[-1])
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            
            self.env = Trajectory(self.path)
            length=len(self.env)
            self._keys =list(range(length))
            self.num_samples = length
            

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if sharding, remap idx to appropriate idx of the sharded set
        
        if idx in self.cached:
            return self.cached[idx]
        
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            ats=self.envs[db_idx][el_idx]
            data_object=self.atoms_to_data(ats)
        else:
            ats = self.env[idx]
            data_object=self.atoms_to_data(ats)
            
        self.cached[idx]=data_object

        return data_object
    
    def atoms_to_data(self,ats):
        data_object = self.a2g.convert(ats)
        if "energies" in ats.calc.results:
            data_object.energies=torch.Tensor(ats.calc.results["energies"].reshape(-1,1))
        if self.a2g.r_edges:
            data_object.neighbors=data_object.cell_offsets.shape[0]
            
        return data_object
    
    
    def close(self):
        # if sharding, remap idx to appropriate idx of the sharded set
        
        
        if not self.path.is_file():
            
            for e in self.envs:
                e.close()
        else:
            self.env.close()

    