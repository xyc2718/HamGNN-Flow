'''
Descripttion: The script to calculat bands from the results of HamGNN
version: 1.0
Author: Yang Zhong
Date: 2022-12-20 14:08:52
LastEditors: Yang Zhong
LastEditTime: 2024-08-06 10:53:04
'''

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.core.periodic_table import Element
import math
import os
import argparse
import yaml
import torch
import logging
import importlib.util

utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
kpoints_generator = utils_module.kpoints_generator
for name in dir(utils_module):
    if not name.startswith('_'):
        globals()[name] = getattr(utils_module, name)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def band_cal(input,*args, **kwargs):
    ################################ Input parameters begin ####################
    yrange=input.get('yrange', (-5, 5)) # The range of y axis in the band structure plot
    nao_max = input["nao_max"] # The maximum number of atomic orbitals, can be 14, 19 or 26
    graph_data_path = input["graph_data_path"] # The path to the graph data file, which is a npz file containing the graph data
    hamiltonian_path = input['hamiltonian_path']
    nk = input['nk']          # the number of k points
    save_dir = input['save_dir'] # The directory to save the results
    filename = input['strcture_name']  # The name of each cif file saved is filename_idx.cif after band calculation band from graph_data.npz
    dpi= input.get('dpi', 300) # The dpi of the saved band structure plot
    save_fig = input.get('save_fig', True) # Whether to save the band structure plot

    # Ham_type
    if 'Ham_type' in input:
        Ham_type = input['Ham_type'].lower()
    else:
        Ham_type = 'openmx'
    
    # soc_switch
    if 'soc_switch' in input:
        soc_switch = input['soc_switch']
    else:
        soc_switch = False
    
    # spin_colinear
    if 'spin_colinear' in input:
        spin_colinear = input['spin_colinear']
    else:
        spin_colinear = False
    
    auto_mode = input['auto_mode']
    if not auto_mode:
        k_path=input['k_path'] 
        label=input['label'] 
    ################################ Input parameters end ######################
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())
    is_single_graph = (len(graph_dataset) == 1)

    num_val = np.zeros((99,), dtype=int)
    if Ham_type == 'openmx':
        for k in num_valence_openmx.keys():
            num_val[k] = num_valence_openmx[k]
    elif Ham_type == 'abacus':
        for k in num_valence_abacus.keys():
            num_val[k] = num_valence_abacus[k]
    else:
        raise NotImplementedError

    # parse the Atomic Orbital Basis Sets
    basis_definition = np.zeros((99, nao_max))
    # key is the atomic number, value is the index of the occupied orbits.
    if Ham_type == 'openmx':
        if nao_max == 14:
            basis_def = basis_def_14
        elif nao_max == 19:
            basis_def = basis_def_19
        else:
            basis_def = basis_def_26
    elif Ham_type == 'abacus':
        if nao_max == 27:
            basis_def = basis_def_27_abacus
        elif nao_max == 40:
            basis_def = basis_def_40_abacus
        else:
            raise NotImplementedError     
    else:
        raise NotImplementedError

    for k in basis_def.keys():
        basis_definition[k][basis_def[k]] = 1
    
    if soc_switch:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(2*(len(graph_dataset[i].Hon)+len(graph_dataset[i].Hoff)))
    
        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hsoc_all = []
            idx = 0
            for i in range(0, len(len_H)):
                Hsoc_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
        else:
            Hsoc_all = []
            for data in graph_dataset:
                Hsoc_all.append(torch.cat([data.Hon, data.Hoff, data.iHon, data.iHoff], dim=0).numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hsoc = Hsoc_all[idx].reshape(-1, 2*nao_max, 2*nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            cell_shift = data.cell_shift.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            if is_single_graph:
                struct.to(filename=os.path.join(save_dir, filename+'.cif'))
            else:
                struct.to(filename=os.path.join(save_dir, filename+f'_{idx+1}.cif'))

            # Initialize k_path and label
            # -------------------- 基于坐标的去重逻辑 --------------------
            if auto_mode:
                kpath_seek = KPathSeek(structure=struct)
                original_klabels = []
                for lbs in kpath_seek.kpath['path']:
                    original_klabels.extend(lbs)
                logging.debug(f'Original klabels from Pymatgen: {original_klabels}')

                if not original_klabels:
                    raise ValueError("K-path generation failed, no labels found.")

                # --- 新的去重逻辑：基于坐标 ---
                final_klabels = [original_klabels[0]]
                final_k_path = [kpath_seek.kpath['kpoints'][original_klabels[0]]]

                for k_label in original_klabels[1:]:
                    current_k_coords = np.array(kpath_seek.kpath['kpoints'][k_label])
                    last_k_coords = np.array(final_k_path[-1])
                    logging.debug(f'Checking k-label: {k_label}, current coords: {current_k_coords}, last coords: {last_k_coords}')
                    # 使用 np.allclose 检查坐标是否几乎完全相同
                    if not np.allclose(current_k_coords, last_k_coords):
                        final_klabels.append(k_label)
                        final_k_path.append(current_k_coords.tolist())

                k_path = final_k_path
                label = [rf'${lb}$' for lb in final_klabels]
                logging.debug(f'Deduplicated k-path labels: {final_klabels}')
        
            Hsoc_real, Hsoc_imag = np.split(Hsoc, 2, axis=0)
            Hsoc = [Hsoc_real[:, :nao_max, :nao_max]+1.0j*Hsoc_imag[:, :nao_max, :nao_max], 
                    Hsoc_real[:, :nao_max, nao_max:]+1.0j*Hsoc_imag[:, :nao_max, nao_max:], 
                    Hsoc_real[:, nao_max:, :nao_max]+1.0j*Hsoc_imag[:, nao_max:, :nao_max],
                    Hsoc_real[:, nao_max:, nao_max:]+1.0j*Hsoc_imag[:, nao_max:, nao_max:]]
    
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
            
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
    
            # cell index
            cell_shift_tuple = [tuple(c) for c in cell_shift.tolist()] # len: (nedges,)
            cell_shift_set = set(cell_shift_tuple)
            cell_shift_list = list(cell_shift_set)
            cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple] # len: (nedges,)
            ncells = len(cell_shift_set)
    
            # SK
            natoms = len(species)
            eigen = []
            for ik in range(nk):
                phase = np.zeros((ncells,),dtype=np.complex64) # shape (ncells,)
                phase[cell_index] = np.exp(2j*np.pi*np.sum(nbr_shift[:,:]*k_vec[ik,None,:], axis=-1))    
                na = np.arange(natoms)
        
                S_cell = np.zeros((ncells, natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                S_cell[cell_index, edge_index[0], edge_index[1], :, :] = Soff  
        
                SK = np.einsum('ijklm, i->jklm', S_cell, phase) # (natoms, natoms, nao_max, nao_max)
                SK[na,na,:,:] +=  Son[na,:,:]
                SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                SK = SK.reshape(natoms*nao_max, natoms*nao_max)
                SK = SK[orb_mask > 0]
                norbs = int(math.sqrt(SK.size))
                SK = SK.reshape(norbs, norbs)
                I = np.identity(2,dtype=np.complex64)
                SK = np.kron(I,SK)
    
                HK_list = []
                for H in Hsoc:
                    Hon = H[:natoms,:,:]
                    Hoff = H[natoms:,:,:] 
                    H_cell = np.zeros((ncells, natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                    H_cell[cell_index, edge_index[0], edge_index[1], :, :] = Hoff    
        
                    HK = np.einsum('ijklm, i->jklm', H_cell, phase) # (natoms, natoms, nao_max, nao_max)
                    HK[na,na,:,:] +=  Hon[na,:,:] # shape (nk, natoms, nao_max, nao_max)
        
                    HK = np.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
                    HK = HK.reshape(natoms*nao_max, natoms*nao_max)
        
                    # mask HK
                    HK = HK[orb_mask > 0]
                    norbs = int(math.sqrt(HK.size))
                    HK = HK.reshape(norbs, norbs)
        
                    HK_list.append(HK)
        
                HK = np.block([[HK_list[0],HK_list[1]],[HK_list[2],HK_list[3]]])
            
                SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                L = torch.linalg.cholesky(SK_cuda)
                L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                L_inv = torch.linalg.inv(L)
                L_t_inv = torch.linalg.inv(L_t)
                Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                orbital_energies, _ = torch.linalg.eigh(Hs)
                orbital_energies = orbital_energies.squeeze(0)
                eigen.append(orbital_energies.cpu().numpy())
            
            eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)
    
            # plot fermi line    
            num_electrons = np.sum(num_val[species])
            max_val = np.max(eigen[num_electrons-1])
            min_con = np.min(eigen[num_electrons])
            eigen = eigen - max_val
            print(f"max_val = {max_val} eV")
            print(f"band gap = {min_con - max_val} eV")
            if nk > 1 and save_fig:
                # plotting of band structure
                print('Plotting bandstructure...')
            
                # First make a figure object
                fig, ax = plt.subplots()
                try:
                
                    # specify horizontal axis details
                    ax.set_xlim(k_node[0],k_node[-1])
                    ax.set_xticks(k_node)
                    ax.set_xticklabels(label)
                    for n in range(len(k_node)):
                        ax.axvline(x=k_node[n], linewidth=0.5, color='k')
                
                    # plot bands
                    for n in range(norbs):
                        ax.plot(k_dist, eigen[n])
                    ax.plot(k_dist, nk*[0.0], linestyle='--')
                
                    # put title
                    ax.set_title("Band structure")
                    ax.set_xlabel("Path in k-space")
                    ax.set_ylabel("Band energy (eV)")
                    ax.set_ylim(yrange)
                    # make an PDF figure of a plot
                    # fig.tight_layout()
                    if is_single_graph:
                        plt.savefig(os.path.join(save_dir, 'band.png'),dpi=dpi)
                    else:
                        plt.savefig(os.path.join(save_dir, f'band_{idx+1}.png'),dpi=dpi)#保存图片
                    print('Done.\n')
                except Exception as e:
                    logging.error(f"Error while plotting band structure: {e}")
                finally:
                    plt.close(fig)
            
            # Export energy band data
            if is_single_graph:
                text_file = open(os.path.join(save_dir, 'band.dat'), "w")
            else:
                text_file = open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w")
        
            text_file.write("# k_lable: ")
            for ik in range(len(label)):
                text_file.write("%s " % label[ik])
            text_file.write("\n")
        
            text_file.write("# k_node: ")
            for ik in range(len(k_node)):
                text_file.write("%f  " % k_node[ik])
            text_file.write("\n")
        
            node_index = node_index[1:]
            for nb in range(len(eigen)):
                for ik in range(nk):
                    text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))
                    if ik in node_index[:-1]:
                        text_file.write('\n')
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))       
                text_file.write('\n')
            text_file.close()

    elif spin_colinear:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(len(graph_dataset[i].Hon))
            len_H.append(len(graph_dataset[i].Hoff))

        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hon_all, Hoff_all = [], []
            idx = 0
            for i in range(0, len(len_H), 2):
                Hon_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
                Hoff_all.append(H[idx:idx + len_H[i+1]])
                idx = idx+len_H[i+1]
        else:
            Hon_all, Hoff_all = [], []
            for data in graph_dataset:
                Hon_all.append(data.Hon.numpy())
                Hoff_all.append(data.Hoff.numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hon = Hon_all[idx].reshape(-1, 2, nao_max, nao_max)
            Hoff = Hoff_all[idx].reshape(-1, 2, nao_max, nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            if is_single_graph:
                struct.to(filename=os.path.join(save_dir, filename+'.cif'))
            else:
                struct.to(filename=os.path.join(save_dir, filename+f'_{idx+1}.cif'))
        
# Initialize k_path and label
            # -------------------- 基于坐标的去重逻辑 --------------------
            if auto_mode:
                kpath_seek = KPathSeek(structure=struct)
                original_klabels = []
                for lbs in kpath_seek.kpath['path']:
                    original_klabels.extend(lbs)
                logging.debug(f'Original klabels from Pymatgen: {original_klabels}')

                if not original_klabels:
                    raise ValueError("K-path generation failed, no labels found.")

                # --- 新的去重逻辑：基于坐标 ---
                final_klabels = [original_klabels[0]]
                final_k_path = [kpath_seek.kpath['kpoints'][original_klabels[0]]]

                for k_label in original_klabels[1:]:
                    current_k_coords = np.array(kpath_seek.kpath['kpoints'][k_label])
                    last_k_coords = np.array(final_k_path[-1])
                    logging.debug(f'Checking k-label: {k_label}, current coords: {current_k_coords}, last coords: {last_k_coords}')
                    # 使用 np.allclose 检查坐标是否几乎完全相同
                    if not np.allclose(current_k_coords, last_k_coords):
                        final_klabels.append(k_label)
                        final_k_path.append(current_k_coords.tolist())
                
                k_path = final_k_path
                label = [rf'${lb}$' for lb in final_klabels]
                logging.debug(f'Deduplicated k-path labels: {final_klabels}')   
                
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
        
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
        
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        
            natoms = len(struct)
            
            for ispin in range(2):
                eigen = []
                for ik in range(len(k_vec)):            
                    HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                    SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
    
                    na = np.arange(natoms)
                    HK[na,na,:,:] +=  Hon[na, ispin, :, :] # shape (natoms, nao_max, nao_max)
                    SK[na,na,:,:] +=  Son[na, :, :]
    
                    coe = np.exp(2j*np.pi*np.sum(nbr_shift*k_vec[ik][None,:], axis=-1)) # shape (nedges,)
    
                    for iedge in range(len(Hoff)):
                        # shape (nao_max, nao_max) += (1, 1)*(nao_max, nao_max)
                        HK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Hoff[iedge,ispin,:,:]
                        SK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Soff[iedge,:,:]
    
    
                    HK = np.swapaxes(HK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                    HK = HK.reshape(natoms*nao_max, natoms*nao_max)
                    SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                    SK = SK.reshape(natoms*nao_max, natoms*nao_max)
    
                    # mask HK and SK
                    #HK = torch.masked_select(HK, orb_mask[idx].repeat(nk,1,1) > 0)
                    HK = HK[orb_mask > 0]
                    norbs = int(math.sqrt(HK.size))
                    HK = HK.reshape(norbs, norbs)
    
                    #SK = torch.masked_select(SK, orb_mask[idx].repeat(nk,1,1) > 0)
                    SK = SK[orb_mask > 0]
                    norbs = int(math.sqrt(SK.size))
                    SK = SK.reshape(norbs, norbs)

                    SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                    HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                    L = torch.linalg.cholesky(SK_cuda)
                    L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                    L_inv = torch.linalg.inv(L)
                    L_t_inv = torch.linalg.inv(L_t)
                    Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                    orbital_energies, _ = torch.linalg.eigh(Hs)
                    orbital_energies = orbital_energies.squeeze(0)
                    eigen.append(orbital_energies.cpu().numpy())

                eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)

                # plot fermi line    
                num_electrons = np.sum(num_val[species])
                max_val = np.max(eigen[math.ceil(num_electrons/2)-1])
                min_con = np.min(eigen[math.ceil(num_electrons/2)])
                eigen = eigen - max_val
                print(f'band info for spin No.{ispin}')
                print(f"max_val = {max_val} eV")
                print(f"band gap = {min_con - max_val} eV")

                if nk > 1 and save_fig:
                    # plotting of band structure
                    print('Plotting bandstructure...')

                    # First make a figure object
                    fig, ax = plt.subplots()
                    try:
                        # specify horizontal axis details
                        ax.set_xlim(k_node[0],k_node[-1])
                        ax.set_xticks(k_node)
                        ax.set_xticklabels(label)
                        for n in range(len(k_node)):
                            ax.axvline(x=k_node[n], linewidth=0.5, color='k')

                        # plot bands
                        for n in range(norbs):
                            ax.plot(k_dist, eigen[n])
                        ax.plot(k_dist, nk*[0.0], linestyle='--')

                        # put title
                        ax.set_title("Band structure")
                        ax.set_xlabel("Path in k-space")
                        ax.set_ylabel("Band energy (eV)")
                        ax.set_ylim(yrange)
                        # make an PDF figure of a plot
                        # fig.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'band_spin{ispin}_{idx+1}.png'))#保存图片
                        print('Done.\n')
                    except Exception as e:
                        logging.error(f"Error while plotting band structure: {e}")
                    finally:
                        plt.close(fig)
                # Export energy band data
                text_file = open(os.path.join(save_dir, f'band_spin{ispin}_{idx+1}.dat'), "w")

                text_file.write("# k_lable: ")
                for ik in range(len(label)):
                    text_file.write("%s " % label[ik])
                text_file.write("\n")

                text_file.write("# k_node: ")
                for ik in range(len(k_node)):
                    text_file.write("%f  " % k_node[ik])
                text_file.write("\n")

                node_index = node_index[1:]
                for nb in range(len(eigen)):
                    for ik in range(nk):
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))
                        if ik in node_index[:-1]:
                            text_file.write('\n')
                            text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))       
                    text_file.write('\n')
                text_file.close()
    
    else:
        # Calculate the length of H for each structure
        len_H = []
        for i in range(len(graph_dataset)):
            len_H.append(len(graph_dataset[i].Hon))
            len_H.append(len(graph_dataset[i].Hoff))
               
        if hamiltonian_path is not None:
            H = np.load(hamiltonian_path)
            Hon_all, Hoff_all = [], []
            idx = 0
            for i in range(0, len(len_H), 2):
                Hon_all.append(H[idx:idx + len_H[i]])
                idx = idx+len_H[i]
                Hoff_all.append(H[idx:idx + len_H[i+1]])
                idx = idx+len_H[i+1]
        else:
            Hon_all, Hoff_all = [], []
            for data in graph_dataset:
                Hon_all.append(data.Hon.numpy())
                Hoff_all.append(data.Hoff.numpy())
        
        wfn_all = []
        for idx, data in enumerate(graph_dataset):
            # build crystal structure
            Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
            Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
            Hon = Hon_all[idx].reshape(-1, nao_max, nao_max)
            Hoff = Hoff_all[idx].reshape(-1, nao_max, nao_max)
            latt = data.cell.numpy().reshape(3,3)
            pos = data.pos.numpy()*au2ang
            nbr_shift = data.nbr_shift.numpy()
            edge_index = data.edge_index.numpy()
            species = data.z.numpy()
            struct = Structure(lattice=latt*au2ang, species=[Element.from_Z(k).symbol for k in species], coords=pos, coords_are_cartesian=True)
            if is_single_graph:
                struct.to(filename=os.path.join(save_dir, filename+'.cif'))
            else:
                struct.to(filename=os.path.join(save_dir, filename+f'_{idx+1}.cif'))
        
            # Initialize k_path and label
            # -------------------- 基于坐标的去重逻辑 --------------------
            if auto_mode:
                kpath_seek = KPathSeek(structure=struct)
                original_klabels = []
                for lbs in kpath_seek.kpath['path']:
                    original_klabels.extend(lbs)
                logging.debug(f'Original klabels from Pymatgen: {original_klabels}')

                if not original_klabels:
                    raise ValueError("K-path generation failed, no labels found.")

                # --- 新的去重逻辑：基于坐标 ---
                final_klabels = [original_klabels[0]]
                final_k_path = [kpath_seek.kpath['kpoints'][original_klabels[0]]]

                for k_label in original_klabels[1:]:
                    current_k_coords = np.array(kpath_seek.kpath['kpoints'][k_label])
                    last_k_coords = np.array(final_k_path[-1])
                    logging.debug(f'Checking k-label: {k_label}, current coords: {current_k_coords}, last coords: {last_k_coords}')
                    # 使用 np.allclose 检查坐标是否几乎完全相同
                    if not np.allclose(current_k_coords, last_k_coords):
                        final_klabels.append(k_label)
                        final_k_path.append(current_k_coords.tolist())
                
                k_path = final_k_path
                label = [rf'${lb}$' for lb in final_klabels]
                logging.debug(f'Deduplicated k-path labels: {final_klabels}')
        
            orb_mask = basis_definition[species].reshape(-1) # shape: [natoms*nao_max] 
            orb_mask = orb_mask[:,None] * orb_mask[None,:]       # shape: [natoms*nao_max, natoms*nao_max]
        
            kpts=kpoints_generator(dim_k=3, lat=latt)
            k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
        
            k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
            k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
        
            natoms = len(struct)
            eigen = []
            for ik in range(nk):
                HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
                SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
            
                na = np.arange(natoms)
                HK[na,na,:,:] +=  Hon[na,:,:] # shape (natoms, nao_max, nao_max)
                SK[na,na,:,:] +=  Son[na,:,:]
            
                coe = np.exp(2j*np.pi*np.sum(nbr_shift*k_vec[ik][None,:], axis=-1)) # shape (nedges,)
            
                for iedge in range(len(Hoff)):
                    # shape (nao_max, nao_max) += (1, 1)*(nao_max, nao_max)
                    HK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Hoff[iedge,:,:]
                    SK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Soff[iedge,:,:]
            
            
                HK = np.swapaxes(HK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                HK = HK.reshape(natoms*nao_max, natoms*nao_max)
                SK = np.swapaxes(SK,-2,-3) #(natoms, nao_max, natoms, nao_max)
                SK = SK.reshape(natoms*nao_max, natoms*nao_max)
            
                # mask HK and SK
                #HK = torch.masked_select(HK, orb_mask[idx].repeat(nk,1,1) > 0)
                HK = HK[orb_mask > 0]
                norbs = int(math.sqrt(HK.size))
                HK = HK.reshape(norbs, norbs)
                        
                #SK = torch.masked_select(SK, orb_mask[idx].repeat(nk,1,1) > 0)
                SK = SK[orb_mask > 0]
                norbs = int(math.sqrt(SK.size))
                SK = SK.reshape(norbs, norbs)

                SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
                HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
                L = torch.linalg.cholesky(SK_cuda)
                L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
                L_inv = torch.linalg.inv(L)
                L_t_inv = torch.linalg.inv(L_t)
                Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
                orbital_energies, _ = torch.linalg.eigh(Hs)
                orbital_energies = orbital_energies.squeeze(0)
                eigen.append(orbital_energies.cpu().numpy())
            
            eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev # (nbands, nk)
            
            # plot fermi line    
            num_electrons = np.sum(num_val[species])
            max_val = np.max(eigen[math.ceil(num_electrons/2)-1])
            min_con = np.min(eigen[math.ceil(num_electrons/2)])
            eigen = eigen - max_val
            print(f"max_val = {max_val} eV")
            print(f"band gap = {min_con - max_val} eV")
            
            if nk > 1 and save_fig and save_fig:
                # plotting of band structure
                print('Plotting bandstructure...')
                try:
                    # First make a figure object
                    fig, ax = plt.subplots()
                
                    # specify horizontal axis details
                    ax.set_xlim(k_node[0],k_node[-1])
                    ax.set_xticks(k_node)
                    ax.set_xticklabels(label)
                    for n in range(len(k_node)):
                        ax.axvline(x=k_node[n], linewidth=0.5, color='k')
                
                    # plot bands
                    for n in range(norbs):
                        ax.plot(k_dist, eigen[n])
                    ax.plot(k_dist, nk*[0.0], linestyle='--')
                
                    # put title
                    ax.set_title("Band structure")
                    ax.set_xlabel("Path in k-space")
                    ax.set_ylabel("Band energy (eV)")
                    ax.set_ylim(yrange)
                    # make an PDF figure of a plot
                    # fig.tight_layout()
                    if is_single_graph:
                        plt.savefig(os.path.join(save_dir, f'band.png'),dpi=dpi)
                    else:
                        plt.savefig(os.path.join(save_dir, f'band_{idx+1}.png'),dpi=dpi)#保存图片
                    print('Done.\n')
                except Exception as e:
                    logging.error(f"Error while plotting band structure: {e}")
                finally:
                    plt.close(fig)
            # Export energy band data
            if is_single_graph:
                text_file = open(os.path.join(save_dir, 'band.dat'), "w")
            else:
                text_file = open(os.path.join(save_dir, f'band_{idx+1}.dat'), "w")
        
            text_file.write("# k_lable: ")
            for ik in range(len(label)):
                text_file.write("%s " % label[ik])
            text_file.write("\n")
        
            text_file.write("# k_node: ")
            for ik in range(len(k_node)):
                text_file.write("%f  " % k_node[ik])
            text_file.write("\n")
        
            node_index = node_index[1:]
            for nb in range(len(eigen)):
                for ik in range(nk):
                    text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))
                    if ik in node_index[:-1]:
                        text_file.write('\n')
                        text_file.write("%f    %f\n" % (k_dist[ik], eigen[nb,ik]))       
                text_file.write('\n')
            text_file.close()

if __name__ == '__main__':
    args= argparse.ArgumentParser(description='Band calculation from HamGNN results')
    args.add_argument('--input', type=str, required=True, help='Path to the input YAML file')
    args = args.parse_args()
    with open(args.input, 'r') as f:
        input = yaml.safe_load(f)
        band_cal(input)