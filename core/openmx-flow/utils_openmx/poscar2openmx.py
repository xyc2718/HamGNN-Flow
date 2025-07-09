__all__ = ['poscar_to_openmxfile', 'main']  # 在文件顶部添加
'''
Descripttion: Script for converting poscar to openmx input file
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:03:36
LastEditors: Yang Zhong
LastEditTime: 2023-07-18 03:24:04
'''

from pymatgen.core.structure import Structure
import glob
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os
import natsort
from utils_openmx.utils import *
import argparse
import yaml
import os
import logging  
from ...utils import get_package_path
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
basic_config_path = os.path.join(parent_dir, 'openmx_basic_config.yaml')
with open(basic_config_path, 'r', encoding='utf-8') as f:
    basic_config = yaml.safe_load(f)
DEFAULT_DATA_PATH = get_package_path("openmx-flow/DFT_DATA19")
DATA_PATH = basic_config.get('DATA_PATH', None) or DEFAULT_DATA_PATH
def poscar_to_openmxfile(structure, system_name="SystemName",filename="openmxDTA.dat",DosKgrid=(4, 4, 4),
                         ScfKgrid=(4, 4, 4),SpinPolarization='off',XcType='GGA-PBE',ElectronicTemperature=100,energycutoff=150,maxIter=1,ScfCriterion=1.0e-6):
    """
    Convert a POSCAR file to an OpenMX input file.
    """
    logging.debug("DATA_PATH_DEFAULT: {}".format(DEFAULT_DATA_PATH))
    logging.debug(f"DATA_PATH: {DATA_PATH}")
    logging.debug(f"Converting {structure} to OpenMX input file {filename}")
    basic_command = f"""
# openmx calculation parameters
  #
  #      File Name      
  #

  System.CurrentDirectory         ./   # default=./
  System.Name                     {system_name}  # default=SystemName
  DATA.PATH           {DATA_PATH}  # default=../openmx/DFT_DATA19
  level.of.stdout                   1    # default=1 (1-3)
  level.of.fileout                  1    # default=1 (0-2)
  HS.fileout                   on       # on|off, default=off

  #
  # SCF or Electronic System
  #

  scf.XcType                  {XcType}    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
  scf.SpinPolarization        {SpinPolarization}        # On|Off|NC
  scf.ElectronicTemperature  {ElectronicTemperature}       # default=300 (K)
  scf.energycutoff           {energycutoff}       # default=150 (Ry)
  scf.maxIter                 {1}         # default=40
  scf.EigenvalueSolver        Band      # DC|GDC|Cluster|Band
  scf.Kgrid                  {ScfKgrid[0]} {ScfKgrid[1]} {ScfKgrid[2]}       # means 4x4x4
  scf.Mixing.Type           rmm-diis     # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
  scf.Init.Mixing.Weight     0.10        # default=0.30 
  scf.Min.Mixing.Weight      0.001       # default=0.001 
  scf.Max.Mixing.Weight      0.400       # default=0.40 
  scf.Mixing.History          7          # default=5
  scf.Mixing.StartPulay       5          # default=6
  scf.criterion             {ScfCriterion}      # default=1.0e-6 (Hartree)

  #
  # MD or Geometry Optimization
  #

  MD.Type                      Nomd        # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                         # Constraint_Opt|DIIS2|Constraint_DIIS2
  MD.Opt.DIIS.History          4
  MD.Opt.StartDIIS             5         # default=5
  MD.maxIter                 100         # default=1
  MD.TimeStep                1.0         # default=0.5 (fs)
  MD.Opt.criterion          1.0e-4       # default=1.0e-4 (Hartree/bohr)

  #
  # MO output
  #

  MO.fileout                  off        # on|off, default=off
  num.HOMOs                    2         # default=1
  num.LUMOs                    2         # default=1

  #
  # DOS and PDOS
  #

  Dos.fileout                  off       # on|off, default=off
  Dos.Erange              -10.0  10.0    # default = -20 20 
  Dos.Kgrid                 {DosKgrid[0]} {DosKgrid[1]} {DosKgrid[2]}      # default = Kgrid1 Kgrid2 Kgrid3
""" 
    try:
        crystal = Structure.from_file(structure)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell = ase_atoms.get_cell().array
    except Exception as e:
        raise(
            ValueError(f"Error reading structure from {structure}: {e}")
        )
    logging.debug(f"Successfully read structure from {crystal}")
    try:
        ase_atoms_to_openmxfile(ase_atoms, basic_command, spin_set, PAO_dict, PBE_dict, filename)
    except Exception as e:
        raise(
            ValueError(f"Error writing OpenMX file {filename}: {e}")
        )


def main(args):    
    with open(args.config, encoding='utf-8') as rstream:
        input = yaml.load(rstream, yaml.SafeLoader)
    
    system_name = input['system_name']
    poscar_path = input['poscar_path'] # The path of poscar or cif files
    filepath = input['filepath'] # openmx file directory to save
    basic_command = input['basic_command']
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    f_vasp = glob.glob(poscar_path) # poscar or cif file directory
    f_vasp = natsort.natsorted(f_vasp)

    for i, poscar in enumerate(f_vasp):
        cif_id = str(i+1)
        crystal = Structure.from_file(poscar)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell = ase_atoms.get_cell().array
        filename =  os.path.join(filepath, f'{system_name}_'+ cif_id + ".dat")
        ase_atoms_to_openmxfile(ase_atoms, basic_command, spin_set, PAO_dict, PBE_dict, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openmx dat file generation')
    parser.add_argument('--config', default='poscar2openmx.yaml', type=str, metavar='N')
    args = parser.parse_args()
    main(args)