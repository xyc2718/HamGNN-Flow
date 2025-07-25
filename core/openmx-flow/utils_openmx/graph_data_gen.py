'''
Descripttion: The scripts used to generate the input file graph_data.npz for HamGNN.
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:07:54
LastEditors: Yang Zhong
LastEditTime: 2023-12-18 16:13:22
'''

import json
import numpy as np
import os
import sys
from torch_geometric.data import Data
import torch
import glob
import natsort
from tqdm import tqdm
import re
from pymatgen.core.periodic_table import Element
from utils_openmx.utils import *
import argparse
import yaml
import logging
from pathlib import Path
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
basic_config_path = parent_dir / 'openmx_basic_config.yaml'
with open(basic_config_path, 'r', encoding='utf-8') as f:
    basic_config = yaml.safe_load(f)
read_openmx_default = parent_dir.parent / "bin/read_openmx"
read_openmx_path = basic_config.get('read_openmx_path', None) or read_openmx_default
read_openmx_path = str(read_openmx_path)
max_SCF_skip = 200

logging.debug(f"read_openmx_path: {str(read_openmx_path)}")
def graph_data_gen(input):
    nao_max = input['nao_max']
    graph_data_path = input['graph_data_save_path']
    # scfout_paths = input['scfout_paths'] # The directory of the .scfout file calculated by openmx/openmx_postprocess, or a wildcard directory name to match multiple directories
    scf_path = input["scf_path"]
    system_name= input['system_name'] # The system name, used to get scf results if scf applied
    isscf = input.get('ifscf', False) 
    dat_file_name = input['dat_file_name']
    if isscf:
        std_file_name =system_name+".std" # None if no openmx computation is performed
        scfout_file_name = system_name+".scfout" # If the openmx self-consistent Hamiltonian is not required as the target, "overlap.scfout" can be used instead.
    else:
        std_file_name = None
        scfout_file_name="overlap.scfout"
    soc_switch = input['soc_switch'] # generate graph_data.npz for SOC (True) or Non-SOC (False) Hamiltonian
    if 'doping_charge' in input:
        doping_charge = input['doping_charge']
    else:
        doping_charge = 0.0
    ################################ Input parameters end ####################

    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    elif nao_max == 26:
        basis_def = basis_def_26
    else:
        raise NotImplementedError
    
    graphs = dict()
    # if not os.path.exists(graph_data_path):
    #     os.makedirs(graph_data_path)
    # scfout_paths = glob.glob(scfout_paths)
    # scfout_paths = natsort.natsorted(scfout_paths)
    logging.debug(f"Processing {scf_path} ...")
    # file paths
    f_sc = os.path.join(scf_path, scfout_file_name)
    f_dat = os.path.join(scf_path, dat_file_name)
    f_H0 = os.path.join(scf_path, "overlap.scfout")

    logging.debug(f"\ndat path: {f_dat}")
    logging.debug(f"\nH0path: {f_H0}")
    logging.debug(f"Scfpath: {f_sc}")

    logging.debug(f"std path: {std_file_name}")
    # read energy
    if std_file_name is not None: 
        logging.debug(f"triggered to read std file: {std_file_name}")  
        f_std = os.path.join(scf_path, std_file_name) 
        try:
            with open(f_std, 'r') as f:
                content = f.read()
                Enpy = float(pattern_eng.findall((content).strip())[0][-1])
                max_SCF = int(pattern_md.findall((content).strip())[-1][-1])
        except:
            raise(ValueError(f"Failed to read energy from {f_std}. "))
    else:
        Enpy = 0.0
        max_SCF = 1    
    # # check if the calculation is converged
    # logging.debug(f"Energy: {Enpy}, max_SCF: {max_SCF}")
    # if max_SCF > max_SCF_skip:
    #     continue  
    # Read crystal parameters
    try:
        with open(f_dat,'r') as f:
            content = f.read()
            speciesAndCoordinates = pattern_coor.findall((content).strip())
            latt = pattern_latt.findall((content).strip())[0]
            latt = np.array([float(var) for var in latt]).reshape(-1, 3)/au2ang
    
            species = []
            coordinates = []
            for item in speciesAndCoordinates:
                species.append(item[0])
                coordinates += item[1:]
            z = atomic_numbers = np.array([Element[s].Z for s in species])
            coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    except:
        raise(ValueError(f"Failed to read crystal parameters from {f_dat}. "))
    
    if soc_switch:
        os.system("cd " + scf_path)
        # read hopping parameters
        os.system(read_openmx_path + " " + f_sc)
        if not os.path.exists(os.path.join(scf_path, "HS.json")):
            raise(ValueError(f'{f_sc} Hs is not found!'))
        
        with open(os.path.join(scf_path, "HS.json"),'r') as load_f:
            try:
                load_dict = json.load(load_f)
            except:
                raise(ValueError(f'{f_sc} Hs is not read successfully!'))
            pos = np.array(load_dict['pos'])
            edge_index = np.array(load_dict['edge_index'])
            inv_edge_idx = np.array(load_dict['inv_edge_idx'])
            #
            Hon = load_dict['Hon']
            Hoff = load_dict['Hoff']
            iHon = load_dict['iHon']
            iHoff = load_dict['iHoff']
            Son = load_dict['Son']
            Soff = load_dict['Soff']
            nbr_shift = np.array(load_dict['nbr_shift'])
            cell_shift = np.array(load_dict['cell_shift'])

            if len(inv_edge_idx) != len(edge_index[0]):
                print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
                quit()

            # Initialize Hks and iHks 
            num_sub_matrix = pos.shape[0] + edge_index.shape[1]
            Hks = np.zeros((num_sub_matrix, 4, nao_max**2))   
            iHks = np.zeros((num_sub_matrix, 3, nao_max**2))
            S = np.zeros((num_sub_matrix, nao_max**2))

            # on-site
            for iatm in range(len(z)):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src = z[iatm]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                S[iatm][mask] = np.array(Son[iatm])
                for i in range(4):
                    Hks[iatm, i][mask] = np.array(Hon[i][iatm]) 
                for i in range(3):
                    iHks[iatm, i][mask] = np.array(iHon[i][iatm])    

            # off-site
            for iedge in range(len(edge_index[0])):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src, tar = z[edge_index[0, iedge]], z[edge_index[1, iedge]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                S[len(z)+iedge][mask] = np.array(Soff[iedge])
                for i in range(4):
                    Hks[iedge + len(z), i][mask] = np.array(Hoff[i][iedge]) 
                for i in range(3):
                    iHks[iedge + len(z), i][mask] = np.array(iHoff[i][iedge])
            #
            hamiltonian_real = np.zeros((num_sub_matrix, 2*nao_max, 2*nao_max)) 
            hamiltonian_real[:,:nao_max,:nao_max] = Hks[:,0,:].reshape(-1, nao_max, nao_max)
            hamiltonian_real[:,:nao_max, nao_max:] = Hks[:,2,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real[:,nao_max:,:nao_max] = Hks[:,2,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real[:,nao_max:,nao_max:] = Hks[:,1,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real = hamiltonian_real.reshape(-1, (2*nao_max)**2)

            hamiltonian_imag = np.zeros((num_sub_matrix, 2*nao_max, 2*nao_max)) 
            hamiltonian_imag[:,:nao_max,:nao_max] = iHks[:,0,:].reshape(-1, nao_max, nao_max)
            hamiltonian_imag[:,:nao_max, nao_max:] = (Hks[:,3,:] + iHks[:,2,:]).reshape(-1, nao_max, nao_max)
            hamiltonian_imag[:,nao_max:,:nao_max] = -(Hks[:,3,:] + iHks[:,2,:]).reshape(-1, nao_max, nao_max)
            hamiltonian_imag[:,nao_max:,nao_max:] = iHks[:,1,:].reshape(-1, nao_max, nao_max)
            hamiltonian_imag = hamiltonian_imag.reshape(-1, (2*nao_max)**2)
        os.system("rm HS.json")

        # read H0
        os.system(read_openmx_path + " " + f_H0)
        if not os.path.exists(os.path.join(scf_path, "HS.json")):
            raise(ValueError(f'{f_H0} is not found!'))
        
        with open(os.path.join(scf_path, "HS.json"),'r') as load_f:
            try:
                load_dict = json.load(load_f)
                logging.debug(f'{f_H0} is read successfully!')
            except:
                raise(ValueError(f'{f_H0} is not read successfully!'))
            #
            Hon0 = load_dict['Hon']
            Hoff0 = load_dict['Hoff']
            iHon0 = load_dict['iHon']
            iHoff0 = load_dict['iHoff']
            Lon = load_dict['Lon']
            Loff = load_dict['Loff']

            # initialize Hks0 and iHks0
            num_sub_matrix = pos.shape[0] + edge_index.shape[1]
            Hks0 = np.zeros((num_sub_matrix, 4, nao_max**2))   
            iHks0 = np.zeros((num_sub_matrix, 3, nao_max**2))
            L = np.zeros((num_sub_matrix, nao_max**2, 3))

            # on-site
            for iatm in range(len(z)):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src = z[iatm]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                L[iatm][mask] = np.array(Lon[iatm])
                for i in range(4):
                    Hks0[iatm, i][mask] = np.array(Hon0[i][iatm]) 
                for i in range(3):
                    iHks0[iatm, i][mask] = np.array(iHon0[i][iatm])    

            # off-site
            for iedge in range(len(edge_index[0])):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src, tar = z[edge_index[0, iedge]], z[edge_index[1, iedge]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                L[len(z)+iedge][mask] = np.array(Loff[iedge])
                for i in range(4):
                    Hks0[iedge + len(z), i][mask] = np.array(Hoff0[i][iedge]) 
                for i in range(3):
                    iHks0[iedge + len(z), i][mask] = np.array(iHoff0[i][iedge])

            hamiltonian_real0 = np.zeros((num_sub_matrix, 2*nao_max, 2*nao_max)) 
            hamiltonian_real0[:,:nao_max,:nao_max] = Hks0[:,0,:].reshape(-1, nao_max, nao_max)
            hamiltonian_real0[:,:nao_max, nao_max:] = Hks0[:,2,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real0[:,nao_max:,:nao_max] = Hks0[:,2,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real0[:,nao_max:,nao_max:] = Hks0[:,1,:].reshape(-1,nao_max, nao_max)
            hamiltonian_real0 = hamiltonian_real0.reshape(-1, (2*nao_max)**2)

            hamiltonian_imag0 = np.zeros((num_sub_matrix, 2*nao_max, 2*nao_max)) 
            hamiltonian_imag0[:,:nao_max,:nao_max] = iHks0[:,0,:].reshape(-1, nao_max, nao_max)
            hamiltonian_imag0[:,:nao_max, nao_max:] = (Hks0[:,3,:] + iHks0[:,2,:]).reshape(-1, nao_max, nao_max)
            hamiltonian_imag0[:,nao_max:,:nao_max] = -(Hks0[:,3,:] + iHks0[:,2,:]).reshape(-1, nao_max, nao_max)
            hamiltonian_imag0[:,nao_max:,nao_max:] = iHks0[:,1,:].reshape(-1, nao_max, nao_max)
            hamiltonian_imag0 = hamiltonian_imag0.reshape(-1, (2*nao_max)**2)
        os.system("rm HS.json")

        graphs[0] = Data( z=torch.LongTensor(z),
                            cell = torch.Tensor(latt[None,:,:]),
                            total_energy=Enpy,
                            pos=torch.FloatTensor(pos),
                            node_counts=torch.LongTensor([len(z)]),
                            edge_index=torch.LongTensor(edge_index),
                            inv_edge_idx=torch.LongTensor(inv_edge_idx),
                            nbr_shift=torch.FloatTensor(nbr_shift),
                            cell_shift=torch.LongTensor(cell_shift),
                            Hon=torch.FloatTensor(hamiltonian_real[:len(z),:]),
                            Hoff=torch.FloatTensor(hamiltonian_real[len(z):,:]),
                            iHon=torch.FloatTensor(hamiltonian_imag[:len(z),:]),
                            iHoff=torch.FloatTensor(hamiltonian_imag[len(z):,:]),
                            Hon0=torch.FloatTensor(hamiltonian_real0[:len(z),:]),
                            Hoff0=torch.FloatTensor(hamiltonian_real0[len(z):,:]),
                            iHon0=torch.FloatTensor(hamiltonian_imag0[:len(z),:]),
                            iHoff0=torch.FloatTensor(hamiltonian_imag0[len(z):,:]),
                            overlap=torch.FloatTensor(S),
                            Son = torch.FloatTensor(S[:pos.shape[0],:]),
                            Soff = torch.FloatTensor(S[pos.shape[0]:,:]),
                            Lon = torch.FloatTensor(L[:pos.shape[0],:,:]),
                            Loff = torch.FloatTensor(L[pos.shape[0]:,:,:]),
                            doping_charge = torch.FloatTensor([doping_charge]))
    else:            
        # read hopping parameters
        logging.debug(os.path.exists(f_sc))
        os.chdir(scf_path)
        logging.debug(os.getcwd())
        run_state=os.system(read_openmx_path + " " + f_sc)
        logging.debug(f"run {read_openmx_path} {f_sc}, run_state: {run_state}")
        if not os.path.exists(os.path.join(scf_path, "HS.json")):
            raise(ValueError(f'{f_sc} Hs is not found!'))
        
        with open(os.path.join(scf_path, "HS.json"),'r') as load_f:
            try:
                load_dict = json.load(load_f)
            except:
                raise(ValueError(f'{f_sc} Hs is not read successfully!'))
            pos = np.array(load_dict['pos'])
            edge_index = np.array(load_dict['edge_index'])
            inv_edge_idx = np.array(load_dict['inv_edge_idx'])
            #
            Hon = load_dict['Hon'][0]
            Hoff = load_dict['Hoff'][0]
            Son = load_dict['Son']
            Soff = load_dict['Soff']
            nbr_shift = np.array(load_dict['nbr_shift'])
            cell_shift = np.array(load_dict['cell_shift'])
            
            # Find inverse edge_index
            if len(inv_edge_idx) != len(edge_index[0]):
                print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
                sys.exit()
    
            #
            num_sub_matrix = pos.shape[0] + edge_index.shape[1]
            H = np.zeros((num_sub_matrix, nao_max**2))
            S = np.zeros((num_sub_matrix, nao_max**2))
            
            for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src = z[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[i][mask] = np.array(sub_maxtrix_H)
                S[i][mask] = np.array(sub_maxtrix_S)
            
            num = 0
            for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H[num + len(z)][mask] = np.array(sub_maxtrix_H)
                S[num + len(z)][mask] = np.array(sub_maxtrix_S)
                num = num + 1
        os.system("rm HS.json")
        
        # read H0
        os.system(read_openmx_path + " " + f_H0)
        if not os.path.exists(os.path.join(scf_path, "HS.json")):
            raise(ValueError(f'{f_H0} is not found!'))
        
        with open(os.path.join(scf_path, "HS.json"),'r') as load_f:
            try:
                load_dict = json.load(load_f)
            except:
                raise(ValueError(f'{f_H0} is not read successfully!'))
            Hon0 = load_dict['Hon'][0]
            Hoff0 = load_dict['Hoff'][0]
    
            #
            num_sub_matrix = pos.shape[0] + edge_index.shape[1]
            H0 = np.zeros((num_sub_matrix, nao_max**2))
            
            for i, sub_maxtrix_H in enumerate(Hon0):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src = z[i]
                mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H0[i][mask] = np.array(sub_maxtrix_H)
            
            num = 0
            for i, sub_maxtrix_H in enumerate(Hoff0):
                mask = np.zeros((nao_max, nao_max), dtype=int)
                src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
                mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
                mask = (mask > 0).reshape(-1)
                H0[num + len(z)][mask] = np.array(sub_maxtrix_H)
                num = num + 1
        os.system("rm HS.json")
        
        # save in Data
        graphs[0] = Data(z=torch.LongTensor(z),
                            cell = torch.Tensor(latt[None,:,:]),
                            total_energy=Enpy,
                            pos=torch.FloatTensor(pos),
                            node_counts=torch.LongTensor([len(z)]),
                            edge_index=torch.LongTensor(edge_index),
                            inv_edge_idx=torch.LongTensor(inv_edge_idx),
                            nbr_shift=torch.FloatTensor(nbr_shift),
                            cell_shift=torch.LongTensor(cell_shift),
                            hamiltonian=torch.FloatTensor(H),
                            overlap=torch.FloatTensor(S),
                            Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                            Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                            Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                            Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                            Son = torch.FloatTensor(S[:pos.shape[0],:]),
                            Soff = torch.FloatTensor(S[pos.shape[0]:,:]),
                            doping_charge = torch.FloatTensor([doping_charge]))
    if len(graphs) == 0:
        print('No valid data found! Please check the input paths or if the DFT calculations are converged.')
    else:
        # np.savez_compressed(graph_data_path, graph=graphs)
        np.savez(graph_data_path, graph=graphs)
        print('The graph data is saved in %s' % graph_data_path)
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate graph data for HamGNN.")
    argparser.add_argument('--graph_data_save_path', type=str, required=True, help='Path to save the graph data.')
    argparser.add_argument('--scf_path', type=str, required=True, help='Path to the directory containing the .scfout file calculated by OpenMX/OpenMX_postprocess.')
    argparser.add_argument('--dat_file_name', type=str, required=True, help='Name of the .dat file generated by OpenMX.')
    argparser.add_argument('--nao_max', type=int, default=26, choices=[14, 19, 26], help='Maximum number of atomic orbitals (NAO) to consider. Default is 26.')
    argparser.add_argument('--soc_switch', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to generate graph data for SOC (True) or Non-SOC (False) Hamiltonian. Default is False.')
    argparser.add_argument('--doping_charge', type=float, default=0.0, help='Doping charge to be added to the system. Default is 0.0.')
    argparser.add_argument('--system_name', type=str, help='The system name, used to get scf results if scf applied.')
    argparser.add_argument('--ifscf', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use self-consistent Hamiltonian from OpenMX. Default is False.')
    args = argparser.parse_args()
    input = {
        'graph_data_save_path': args.graph_data_save_path,
        'scf_path': args.scf_path,
        'dat_file_name': args.dat_file_name,
        'nao_max': args.nao_max,
        'soc_switch': args.soc_switch,
        'doping_charge': args.doping_charge,
        'system_name': args.system_name,
        'ifscf': args.ifscf
    }
    logging.debug(f"Input parameters: {input}")
    graph_data_gen(input)

