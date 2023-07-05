"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging

import submitit

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)
from ocpmodels.datasets.ase_dataset import AseDataset


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)

from ase.calculators.calculator import Calculator,all_changes
import torch
from typing import *
from ase import Atoms
from torch_geometric.data import Data,Batch
from ocpmodels.preprocessing import AtomsToGraphs
    
class Ocpwarpcalc(Calculator):
    implemented_properties = ["energy", "free_energy", "forces","energies"]
    def __init__(self, net:torch.nn.Module,label="OcpNet",rcut=6.0, **kwargs):
        super().__init__(label=label, **kwargs)
        self.net=net
        self.device=next(self.net.parameters()).device
        import inspect
        print(inspect.getfullargspec(self.net.forward))
        if hasattr(self.net,"return_atom_energy") and getattr(self.net,"return_atom_energy",None) == True :
            self.is_atomic_energy=True
        else:
            self.is_atomic_energy=False
            
        self.a2g = AtomsToGraphs(
        max_neigh=50,
        radius=rcut,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=True,
    )
        
    def calculate(self, atoms=None, properties:List[str] = ["energy", "forces"],system_changes: List[str] = all_changes):
        self.atoms = atoms
        if atoms is not None:
            self.atoms = atoms.copy()
        data=self.newatoms2data(self.atoms).to(self.device)
        # print("self.is_atomic_energy  ",self.is_atomic_energy)
        # print("data: ",data,data.batch)
        if self.is_atomic_energy:
            e,f,es=self.net.forward(data,return_atom_energy=True)
        else:
            e,f=self.net(data)
        e=e.detach().cpu().numpy()[0]
        f=f.detach().cpu().numpy()
        # print(f)
        self.results["energy"]=e
        self.results["forces"]=f
        if self.is_atomic_energy:
            self.results["energies"]=es.detach().cpu().numpy().reshape(-1)
            
    def newatoms2data(self,frame:Atoms):
        # frame.set_pbc(True)
        data_object = self.a2g.convert(frame)
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


if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args(["--mode","train","--config-yml","/home/zln/wp/torchlearn/ocpnew/ocp/projects/cnt/oc20_2m_nequip_map_rpc.yml"])
    config = build_config(args, override_args)
    
    from ocpmodels.datasets.ase_dataset import AseDataset
    a=AseDataset("projects/WSecluster/data_trajs/gen1/datas.traj")
    
    with new_trainer_context(args=args, config=config) as ctx:
        config = ctx.config
        task = ctx.task
        trainer = ctx.trainer
        calc=Ocpwarpcalc(trainer.model.module)

        task.setup(trainer)
        loader=trainer.get_dataloader(a,trainer.get_sampler(a,5,True))
        trainer.train_new_data(loader)
        trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        
