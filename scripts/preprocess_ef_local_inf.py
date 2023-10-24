"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import sys

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, samples, pid,num_works= mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    idx=0
    
    
    total_number=0
    for sample in samples:
        traj_path = sample #os.path.join(args.data_path, f"{xyz_idx}.extxyz")
        assert traj_path.split(".")[-1] == "traj"
        from ase.io.trajectory import Trajectory
        traj_frames = Trajectory(traj_path)
        total_number+=len(traj_frames)
        traj_frames.close()
    
    pbar = tqdm(
        total=total_number,
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for sample in samples:
        #print(sample)
        traj_frames = Trajectory(sample)
        for i in range(0+pid,len(traj_frames),num_works):
            #print(i,len(traj_frames),pid)
            pbar.update(num_works)
            frame=traj_frames[i]
            if len(frame) < 5:
                continue
            try:
                data_object = a2g.convert(frame)
            except Exception as e:
                print(e)
                continue
            # add atom tags
            data_object.tags = torch.LongTensor(frame.get_tags())
            if "tags" not in frame.arrays:
                data_object.tags+=1
            data_object.sid = i
            data_object.fid = i
            # print(data_object)
            if a2g.r_edges:
                data_object.neighbors=data_object.cell_offsets.shape[0]
            
            if "energies" in frame.calc.results:
                data_object.energies=torch.Tensor(frame.calc.results["energies"].reshape(-1,1))
            # subtract off reference energy
            # if args.ref_energy and not args.test_data:
            #     ref_energy = float(frame_log[2])
            #     data_object.y -= ref_energy

            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            idx += 1
            #sampled_ids.append(",".join(frame_log[:2]) + "\n")
            # pbar.update(1)
        traj_frames.close()
    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return idx


def main(args):
    xyz_logs = glob.glob(os.path.join(args.data_path, "*.traj"))

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=5000,
        radius=6,
        r_energy=not args.test_data,
        r_forces=not args.test_data,
        r_fixed=True,
        r_distances=False,
        r_edges=args.get_edges,
    )
    
    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    lmdb_offset=0
    for i in range(2000):
        lmdbpath=os.path.join(args.out_path, "data.%04d.lmdb" % i)
        if not os.path.exists(lmdbpath):
            lmdb_offset=i
            break
    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % (i+lmdb_offset))
        for i in range(args.num_workers)
    ]
    
    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            xyz_logs,
            i,
            args.num_workers
        )
        for i in range(args.num_workers)
    ]
    op = list((pool.imap(write_images_to_lmdb, mp_args)))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to dir containing *.extxyz and *.txt files",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Is data being processed test data?",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
