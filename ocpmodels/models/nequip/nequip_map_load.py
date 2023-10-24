"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the Nequip implementation: https://github.com/mir-group/nequip. License:

---
MIT License

Copyright (c) 2021 The President and Fellows of Harvard College

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from select import select
import torch

try:
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.nn import (
        AtomwiseLinear,
        AtomwiseReduce,
        ConvNetLayer,
        PerSpeciesScaleShift,
        SequentialGraphNetwork,
    )
    from nequip.nn.cutoffs import PolynomialCutoff
    from nequip.nn.embedding import (
        OneHotAtomEncoding,
        RadialBasisEdgeEncoding,
        SphericalHarmonicEdgeAttrs,
    )
    from nequip.nn.radial_basis import BesselBasis
    from nequip.utils import Config, Output, instantiate
except ImportError:
    pass
from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)


@registry.register_model("nequipmapload")
class NequipWrap(nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        load_path='',
        cutoff=4.0,
        regress_forces=False,
        direct_forces=False,
        otf_graph=True,
        use_pbc=False,
        max_neighbors=50,
        **convolution_args,
    ):
        super().__init__()
        
        self.return_atom_energy = True
        
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        
        from nequip.train import Trainer
        self.model,self.model_config = Trainer.load_model_from_training_session(load_path)
        self.model=self.model.func
        self.atom_map=self.model_config['type_names']
        from nequip.data.transforms import TypeMapper
        self.type_mapper=TypeMapper(chemical_symbols=self.atom_map)
        

    @staticmethod
    def convert_ocp(data):
        data = AtomicData(
            pos=data.pos,
            edge_index=data.edge_index,
            edge_cell_shift=data.cell_offsets.float(),
            cell=data.cell,
            #atom_types=data.atomic_numbers.long(),
            atomic_numbers=data.atomic_numbers.long(),
            batch=data.batch,
            edge_vectors=data.edge_vec,
        )
        data = AtomicData.to_AtomicDataDict(data)
        
        return data

    def _forward(self, data):
        data = self.type_mapper(data)
        # print(data["atom_types"])
        return self.model(data)

    @conditional_grad(torch.enable_grad())
    def forward(self, data,return_atom_energy=False):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        if self.use_pbc:
            if self.otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, self.cutoff, self.max_neighbors
                )
            else:
                edge_index = data.edge_index
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=False,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_vec = out["distance_vec"]
        else:
            self.otf_graph = True
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff,
                batch=data.batch,
                max_num_neighbors=self.max_neighbors,
            )
            j, i = edge_index
            edge_vec = data.pos[j] - data.pos[i]
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.edge_vec = edge_vec

        data = self.convert_ocp(data)
        out=self._forward(data)
        energy = out["total_energy"]

        if self.regress_forces and self.direct_forces:
            forces = self.model(data)["per_atom_force"]
            return energy, forces

        if self.regress_forces and not self.direct_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data["pos"],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            if return_atom_energy:
                return energy, forces, out["atomic_energy"]
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
