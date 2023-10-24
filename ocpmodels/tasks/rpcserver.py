"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os

from ocpmodels.common.registry import registry
from ocpmodels.trainers.forces_trainer import ForcesTrainer


from ase.calculators.calculator import Calculator,all_changes
from typing import *
import torch
from torch_geometric.data import Data,Batch
import sys
from ase.atoms import Atoms
import numpy as np
#from ase2ocp import AtomsToGraphs
from ocpmodels.preprocessing import AtomsToGraphs
a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=True,
    )
def newatoms2data(frame:Atoms):
    # frame.set_pbc(True)
    data_object = a2g.convert(frame)
    # add atom tags
    
    data_object.tags = torch.LongTensor(frame.get_tags())
    if "tags" not in frame.arrays:
        data_object.tags+=1
        
    data_object.sid = 0
    data_object.fid = 0
    # subtract off reference energy
    # if args.ref_energy and not args.test_data:
    #     ref_energy = float(frame_log[2])
    #     data_object.y -= ref_energy
    data_object.neighbors=data_object.cell_offsets.shape[0]
    # print(data_object)
    return Batch.from_data_list([data_object])
# from ocpmodels.preprocessing.atoms_to_graphs import 
def atoms2data(atoms:Atoms):
    z=atoms.get_atomic_numbers()
    pos=atoms.get_positions()
    cell=atoms.get_cell().array
    if "tags" in atoms.arrays:
        tags=atoms.arrays['tags']
    else:
        tags=np.zeros(pos.shape[0],dtype=np.int32)+1
    
    d0=Data(pos=torch.tensor(pos).float(),atomic_numbers=torch.tensor(z),cell=torch.tensor(cell.reshape(1,3,3)).float(),natoms=torch.tensor([len(z)]),tags=torch.tensor(tags,dtype=torch.long))
    # d0.pos.requires_grad_()
    return Batch.from_data_list([d0])
    
class Ocpwarpcalc(Calculator):
    implemented_properties = ["energy", "free_energy", "forces","energies"]
    def __init__(self, net:torch.nn.Module,label="OcpNet", **kwargs):
        super().__init__(label=label, **kwargs)
        self.net=net
        self.device=next(self.net.parameters()).device
        import inspect
        print(inspect.getfullargspec(self.net.forward))
        if hasattr(self.net,"return_atom_energy") and getattr(self.net,"return_atom_energy",None) == True :
            self.is_atomic_energy=True
        else:
            self.is_atomic_energy=False
        
    def calculate(self, atoms=None, properties:List[str] = ["energy", "forces"],system_changes: List[str] = all_changes):
        self.atoms = atoms
        if atoms is not None:
            self.atoms = atoms.copy()
        data=newatoms2data(self.atoms).to(self.device)
        # print("self.is_atomic_energy  ",self.is_atomic_energy)
        # print("data: ",data,data.batch)
        if self.is_atomic_energy:
            e,f,es=self.net.forward(data,return_atom_energy=True)
        else:
            e,f=self.net(data)
        e=e.detach().cpu().numpy().reshape(-1)[0]
        f=f.detach().cpu().numpy().reshape(-1,3)
        # print(f)
        self.results["energy"]=e
        self.results["forces"]=f
        if self.is_atomic_energy:
            self.results["energies"]=es.detach().cpu().numpy().reshape(-1)
        
def rpc_server_run(calc,address='tcp://0.0.0.0:18953'):
    from pyrpc import Server
    import traceback
    class HelloRPC(object):
        def hello(self, name):
            return "Hello, %s" % name
        def calc_atoms(self,atoms:Atoms):
            # print(atoms)
            try:
                atoms.set_calculator(calc)
                calc.calculate(atoms)
            except Exception:
                traceback.print_exc()
            return (calc.results)
    try:
        s = Server(HelloRPC(),address=address)

        print("######ready####")
        import zmq
        print("address: ",address,s._socket.getsockopt(zmq.LAST_ENDPOINT))

        s.run()
    except Exception as e:
        print(e)
        raise e
        del s

from .task import BaseTask

@registry.register_task("rpccalc")
class RpcCalcsTask(BaseTask):
    def run(self):
        # assert (
        #     self.trainer.test_loader is not None
        # ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        # results_file = "predictions"
        # self.trainer.predict(
        #     self.trainer.test_loader,
        #     results_file=results_file,
        #     disable_tqdm=self.config.get("hide_eval_progressbar", False),
        # )
        global a2g
        a2g = AtomsToGraphs(
        max_neigh=self.config['model']['max_neighbors'],
        radius=self.config['model']['cutoff'],
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=True,
    )
        
        self.trainer.model.eval()
        print("type: ",type(self.trainer.model.module))
        calc=Ocpwarpcalc(self.trainer.model.module)
        # print(self.config)
        print("max_neighbor: ",self.config['model']['max_neighbors'])
        print("cutoff: ",self.config['model']['cutoff'])
        rpc_server_run(calc,address=self.config.get("address","tcp://0.0.0.0:0"))
        
        

