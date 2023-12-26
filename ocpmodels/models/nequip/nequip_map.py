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

from nequip.model import builder_utils
import torch
from nequip.nn import GraphModuleMixin
from torch_runstats.scatter import scatter

class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """



    def __init__(
        self,
        num_types: int,
        r_max:float,
        min_bond_len={},
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={AtomicDataDict.EDGE_LENGTH_KEY: f"0e"},
            irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"},
        )

        from nequip.nn.cutoffs import PolynomialCutoff
        self.cutoff_func=PolynomialCutoff(r_max,6)
        

        self.per_edge_scales = torch.nn.Parameter(torch.ones(num_types, num_types),requires_grad=False)
        if isinstance(min_bond_len,dict):
            for k,v in min_bond_len:
                type1,type2=[int(i) for i in k.split("-")]
                self.per_edge_scales[type1,type2]=v
                self.per_edge_scales[type2,type1]=v
        else:
            self.per_edge_scales=torch.nn.Parameter(torch.ones(num_types, num_types)*min_bond_len,requires_grad=False)

        print("per_scale: ",self.per_edge_scales)
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        l0=(self.per_edge_scales[
            center_species, neighbor_species
        ])
        
        edge_eng = torch.pow(edge_length/l0,-12)/24.0*l0*self.cutoff_func(edge_length)
        edge_eng=edge_eng.unsqueeze(-1)
        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = data[AtomicDataDict.PER_ATOM_ENERGY_KEY]+atom_eng

        return data



@registry.register_model("nequipmap")
class NequipWrap(nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        num_basis=8,  # number of basis functions used in radial basis
        BesselBasis_trainable=True,  # train bessel weights
        PolynomialCutoff_p=6.0,  # p-exponent used in polynomial cutoff
        # irreps used in hidden layer of output block
        conv_to_output_hidden_irreps_out="8x0e",
        # irreps for the chemical embedding of species
        chemical_embedding_irreps_out="8x0e",
        # irreps used for hidden features. Default is lmax=1, with even and odd parities
        feature_irreps_hidden="8x0o + 8x0e + 8x1o + 8x1e",
        # irreps of the spherical harmonics used for edges. If single integer, full SH up to lmax=that_integer
        irreps_edge_sh="0e + 1o",
        nonlinearity_type="gate",
        num_layers=3,  # number of interaction blocks
        invariant_layers=2,  # number of radial layers
        invariant_neurons=64,  # number of hidden neurons in radial function
        use_sc=True,  # use self-connection or not
        cutoff=4.0,
        add_repulsive=False,
        min_bond_len={},
        resnet=False,
        regress_forces=False,
        direct_forces=False,
        otf_graph=True,
        use_pbc=False,
        max_neighbors=50,
        ave_num_neighbors=None,
        atom_map=["H"],
        add_per_species_rescale=False,  # per species/atom scaling
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
        self.ave_num_neighbors = ave_num_neighbors
        self.add_per_species_rescale = add_per_species_rescale
        self.atom_map = atom_map
        # print("atommap: ",self.atom_map)
        config = {
            "BesselBasis_trainable": BesselBasis_trainable,
            "PolynomialCutoff_p": PolynomialCutoff_p,
            "conv_to_output_hidden_irreps_out": conv_to_output_hidden_irreps_out,
            "chemical_embedding_irreps_out": chemical_embedding_irreps_out,
            "feature_irreps_hidden": feature_irreps_hidden,
            "irreps_edge_sh": irreps_edge_sh,
            "nonlinearity_type": nonlinearity_type,
            "num_basis": num_basis,
            "num_layers": num_layers,
            "r_max": cutoff,
            "resnet": resnet,
            "regress_forces": regress_forces,
            "num_types": len(self.atom_map),
            "chemical_symbols": self.atom_map,
            "invariant_layers": invariant_layers,
            "invariant_neurons": invariant_neurons,
            "use_sc": use_sc,
            "avg_num_neighbors": self.ave_num_neighbors,
            "min_bond_len":min_bond_len,
            **convolution_args,
        }
        from nequip.data.transforms import TypeMapper
        self.type_mapper=TypeMapper(chemical_symbols=self.atom_map)
        layers = {
            # -- Encode --
            "one_hot": OneHotAtomEncoding,
            "spharm_edges": SphericalHarmonicEdgeAttrs,
            "radial_basis": RadialBasisEdgeEncoding,
            # -- Embed features --
            "chemical_embedding": AtomwiseLinear,
        }

        # add convnet layers
        # insertion preserves order
        for layer_i in range(num_layers):
            layers[f"layer{layer_i}_convnet"] = ConvNetLayer

        layers.update(
            {
                # -- output block --
                "conv_to_output_hidden": AtomwiseLinear,
                "output_hidden_to_scalar": (
                    AtomwiseLinear,
                    dict(
                        irreps_out="1x0e",
                        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    ),
                ),
            }
        )

        if self.direct_forces:
            layers["output_hidden_to_vect"] = (
                # add direct force to output block
                AtomwiseLinear,
                dict(
                    irreps_out="1x1o",
                    out_field="per_atom_force",
                ),
            )

        if add_per_species_rescale:
            # type_names must be set in the config
            layers["per_species_rescale"] = (
                PerSpeciesScaleShift,
                dict(
                    field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                ),
            )
        if add_repulsive:
            layers["add_vdw_repusive"]=EdgewiseEnergySum

        layers["total_energy_sum"] = (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        )

        self.model = SequentialGraphNetwork.from_parameters(
            shared_params=config, layers=layers
        )
        print("####################config############################")
        # for k,v in config.items():
        #     print(f"{k}: {v}")
        print("config: ",config)
        print("###"*20)
        import yaml,io
        tmpio=io.StringIO()
        yaml.dump(config,tmpio)
        print(tmpio.getvalue())
        print("###"*20)
        print("####################config############################")

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
