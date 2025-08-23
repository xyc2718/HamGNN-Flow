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
import argparse
import yaml
import os
import logging  

from typing import Optional
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import importlib.util
utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
# ase_atoms_to_openmxfile = utils_module.ase_atoms_to_openmxfile
for name in dir(utils_module):
    if not name.startswith('_'):
        globals()[name] = getattr(utils_module, name)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
basic_config_path = os.path.join(parent_dir, 'openmx_basic_config.yaml')
with open(basic_config_path, 'r', encoding='utf-8') as f:
    basic_config = yaml.safe_load(f)
DEFAULT_DATA_PATH = os.path.join(parent_dir,"DFT_DATA19")
DATA_PATH = basic_config.get('DATA_PATH', None) or DEFAULT_DATA_PATH
def poscar_to_openmxfile(structure, system_name="SystemName",filename="openmxDTA.dat",DosKgrid=(4, 4, 4),
                         ScfKgrid=(4, 4, 4),SpinPolarization='off',XcType='GGA-PBE',ElectronicTemperature=100,energycutoff=150,maxIter=1,ScfCriterion=1.0e-6,charge=0.0,type='Nomd',MDmaxIter=100,MDTimeStep=0.5,MDOptcriterion=1.0e-4,mdfile=None):
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
  postprocess       1
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
  scf.maxIter                 {maxIter}         # default=40
  scf.EigenvalueSolver        Band      # DC|GDC|Cluster|Band
  scf.Kgrid                  {ScfKgrid[0]} {ScfKgrid[1]} {ScfKgrid[2]}       # means 4x4x4
  scf.Mixing.Type           rmm-diis     # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
  scf.Init.Mixing.Weight     0.10        # default=0.30 
  scf.Min.Mixing.Weight      0.001       # default=0.001 
  scf.Max.Mixing.Weight      0.400       # default=0.40 
  scf.Mixing.History          7          # default=5
  scf.Mixing.StartPulay       5          # default=6
  scf.criterion             {ScfCriterion}      # default=1.0e-6 (Hartree)
  scf.system.charge          {charge}       # default=0.0

  #
  # MD or Geometry Optimization
  #

  MD.Type                      {type}        # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                         # Constraint_Opt|DIIS2|Constraint_DIIS2
  MD.Opt.DIIS.History          4
  MD.Opt.StartDIIS             5         # default=5
  MD.maxIter                 {MDmaxIter}         # default=1
  MD.TimeStep                {MDTimeStep}         # default=0.5 (fs)
  MD.Opt.criterion          {MDOptcriterion}       # default=1.0e-4 (Hartree/bohr)

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
        if mdfile is not None:
                crystal = update_structure_from_md2(crystal, mdfile)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        # cell = ase_atoms.get_cell().array
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


def parse_last_frame_from_md2(filepath: str) -> Optional[np.ndarray]:
    """
    解析一个类XYZ格式的MD轨迹文件，并返回最后一帧的笛卡尔坐标。
    
    Args:
        filepath (str): .md2 文件的路径。

    Returns:
        Optional[np.ndarray]: 一个包含所有原子笛卡尔坐标的numpy数组，
                              如果解析失败则返回 None。
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到坐标文件 '{filepath}'")
        return None

    coords = []
    num_atoms = 0
    start_index = -1
    # 从文件末尾向前查找，以找到最后一帧的起始位置
    for i in range(len(lines) - 1, -1, -1):
        try:
            line_content = lines[i].strip()
            if len(line_content.split()) == 1 and line_content.isdigit():
                num_atoms = int(line_content)
                # 帧的坐标开始于原子数行之后两行
                start_index = i + 2
                break
        except (ValueError, IndexError):
            continue
    
    if start_index == -1 or num_atoms == 0:
        print(f"错误：无法在 '{filepath}' 中解析出原子数和坐标。")
        return None

    # 提取最后一帧的坐标
    frame_lines = lines[start_index : start_index + num_atoms]
    if len(frame_lines) < num_atoms:
        print(f"错误：文件 '{filepath}' 不完整，最后一帧的原子数少于声明的数量。")
        return None
        
    for line in frame_lines:
        parts = line.split()
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
        
    return np.array(coords)


def update_structure_from_md2(initial_structure: Structure , md2_path: str) -> Optional[Structure]:
    """
    从一个初始结构文件和一个包含最终坐标的.md2文件创建一个新的Pymatgen Structure对象。

    Args:
        initial_structure (Structure): 包含正确晶格信息的初始结构对象。
        md2_path (str): 包含最终原子笛卡尔坐标的 .md2 文件路径。

    Returns:
        Optional[Structure]: 一个更新了坐标的新的Pymatgen Structure对象，
                             如果过程中发生错误则返回 None。
    """
    # 2. 从.md2文件解析新的笛卡尔坐标
    print(f"正在从 '{md2_path}' 解析最终坐标...")
    new_cart_coords = parse_last_frame_from_md2(md2_path)
    
    if new_cart_coords is None:
        return None # 解析失败

    # 3. 验证原子数量是否匹配
    if len(new_cart_coords) != len(initial_structure):
        print("!!! 严重错误 !!!")
        raise ValueError(
            f"初始结构中的原子数 ({len(initial_structure)}) 与 .md2 文件中的原子数 ({len(new_cart_coords)}) 不匹配。"
        )
    
    print(f"坐标解析成功，共 {len(new_cart_coords)} 个原子。")

    # 4. 创建一个新的Structure对象，使用旧的晶格/种类和新的坐标
    updated_structure = Structure(
        lattice=initial_structure.lattice,
        species=initial_structure.species,
        coords=new_cart_coords,
        coords_are_cartesian=True  # 明确告知pymatgen我们提供的是笛卡尔坐标
    )
    print("已成功创建更新后的结构对象。")
    
    return updated_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openmx dat file generation')
    parser.add_argument('--structure', type=str, required=True, help='Structure file path (e.g., POSCAR)')
    parser.add_argument('--system_name', type=str, default="SystemName", help='System name for OpenMX input file')
    parser.add_argument('--filename', type=str, default="openmxDTA.dat", help='Output OpenMX input file name')
    parser.add_argument('--DosKgrid', type=int, nargs=3, default=(4, 4, 4), help='K-point grid for DOS calculation')
    parser.add_argument('--ScfKgrid', type=int, nargs=3, default=(4, 4, 4), help='K-point grid for SCF calculation')
    parser.add_argument('--SpinPolarization', type=str, default='off', help='Spin polarization (on/off)')
    parser.add_argument('--XcType', type=str, default='GGA-PBE', help='Exchange-correlation functional type')
    parser.add_argument('--ElectronicTemperature', type=float, default=100, help='Electronic temperature (K)')
    parser.add_argument('--energycutoff', type=float, default=150, help='Energy cutoff (Ry)')
    parser.add_argument('--maxIter', type=int, default=1, help='Maximum number  of SCF iterations')
    parser.add_argument('--ScfCriterion', type=float, default=1.0e-6, help='SCF convergence criterion (Hartree)')
    parser.add_argument('--charge', type=float, default=0.0, help='Total charge of the system')
    parser.add_argument('--type', type=str, default='Nomd', help='MD type (Nomd/Opt/NVE/NVT_VS/NVT_NH)')
    parser.add_argument('--mdfile', type=str, default=None, help='MD file path (optional)')
    parser.add_argument('--MDmaxIter', type=int, default=100, help='Maximum number of MD iterations')
    parser.add_argument('--MDTimeStep', type=float, default=0.5, help='MD time step (fs)')
    parser.add_argument('--MDOptcriterion', type=float, default=1.0e-4, help='MD optimization criterion (Hartree/bohr)')

    args = parser.parse_args()
    poscar_to_openmxfile(
        structure=args.structure,
        system_name=args.system_name,
        filename=args.filename,
        DosKgrid=args.DosKgrid,
        ScfKgrid=args.ScfKgrid,
        SpinPolarization=args.SpinPolarization,
        XcType=args.XcType,
        ElectronicTemperature=args.ElectronicTemperature,
        energycutoff=args.energycutoff,
        maxIter=args.maxIter,
        ScfCriterion=args.ScfCriterion,
        charge=args.charge,
        type=args.type,
        mdfile=args.mdfile,
        MDmaxIter=args.MDmaxIter,
        MDTimeStep=args.MDTimeStep,
        MDOptcriterion=args.MDOptcriterion
    )