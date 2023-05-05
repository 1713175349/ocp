import logging

from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from pyrpc import Client
from typing import List
class Rpcrpcalc(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]
    def __init__(self, addrc="tcp://0.0.0.0:18953",label="rpcclc", **kwargs):
        super().__init__(label=label, **kwargs)
        self.proxy=Client(address=addrc)
        
        
    def calculate(self, atoms=None, properties:List[str] = ["energy", "forces"],system_changes: List[str] = all_changes):
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # print(f)
        # print(self.atoms)
        self.results=self.proxy.calc_atoms(self.atoms)
        
        
from ase.calculators.singlepoint import SinglePointCalculator


def main():
    import argparse
    parse=argparse.ArgumentParser(description='利用机器学习势能预测分子结构的能量和力')
    parse.add_argument("inputfile",help="input file containing the data")
    parse.add_argument("outputfile",help="输出结果的文件")
    parse.add_argument("--server",'-s',help="计算服务器地址",default="tcp://0.0.0.0:18953")
    parse.add_argument("--index","-i",default=":",help="输入文件的index")
    
    args=parse.parse_args()
    
    clc=Rpcrpcalc(args.server)
    
    from ase.io import read,write
    outs=[]
    try:
        if args.inputfile.endswith(".traj"):
            from ase.io.trajectory import Trajectory
            ats=Trajectory(args.inputfile)
        else:
            ats=read(args.inputfile,index=args.index)
    except Exception as e:
        logging.error(str(e))
        ats=[read(args.inputfile)]
    from tqdm import tqdm
    for a in tqdm(ats):
        clc.calculate(a)
        sclc=SinglePointCalculator(a,**clc.results)
        a.calc=sclc
        outs.append(a)
    
    write(args.outputfile,outs)
    
    
if __name__ == "__main__":
    main()