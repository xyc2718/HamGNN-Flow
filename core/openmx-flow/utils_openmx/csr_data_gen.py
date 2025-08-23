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
import importlib
import argparse
import yaml
import logging
from pathlib import Path
import scipy.sparse as sp
import numpy as np

utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
for name in dir(utils_module):
    if not name.startswith('_'):
        globals()[name] = getattr(utils_module, name)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
au2ang = 0.5291772490000065
au2ev = 27.211324570273
def gen_H_csr(graph, output_file,set_hamiltonian=None,nao_max=26,soc_switch=False):
    """
    Convert graph hamiltonian data to CSR format file
    
    Args:
        graph: graph data containing hamiltonian information
        output_file: path to output CSR file
    """
    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    elif nao_max == 26:
        basis_def = basis_def_26
    else:
        raise NotImplementedError
    
    # Extract data from graph
    z = graph.z.numpy()
    pos = graph.pos.numpy()
    edge_index = graph.edge_index.numpy()
    cell_shift = graph.cell_shift.numpy()
    
    # Calculate total number of orbitals
    total_orbs = sum(len(basis_def[z[i]]) for i in range(len(z)))
    
    # Dictionary to store H(R) matrices
    R_dict = {}
    
    if not soc_switch:  # Non-SOC case
        if set_hamiltonian is not None:
            hamiltonian=set_hamiltonian 
        else:
            hamiltonian = graph.hamiltonian.numpy()
        nao_max = int(np.sqrt(hamiltonian.shape[1]))
        print(f'Total orbitals (without SOC): {total_orbs}, NAO max: {nao_max}')
        # Initialize R = (0,0,0) matrix for on-site terms
        R_dict[(0, 0, 0)] = np.zeros((total_orbs, total_orbs))
        
        # Fill on-site terms
        orbital_offset = 0
        for iatm in range(len(z)):
            src_z = z[iatm]
            src_basis = basis_def[src_z]
            n_orb = len(src_basis)
            
            # Reshape hamiltonian sub-matrix
            H_sub = hamiltonian[iatm].reshape(nao_max, nao_max)

            
            # Extract relevant elements based on basis and convert to eV
            for i in range(n_orb):
                for j in range(n_orb):
                    R_dict[(0, 0, 0)][orbital_offset + i, orbital_offset + j] = H_sub[src_basis[i], src_basis[j]] * au2ev
            orbital_offset += n_orb
        
        # Fill off-site terms
        for iedge in range(edge_index.shape[1]):
            R_vec = tuple(cell_shift[iedge])
            if R_vec not in R_dict:
                R_dict[R_vec] = np.zeros((total_orbs, total_orbs))
            
            src_atom = edge_index[0, iedge]
            tar_atom = edge_index[1, iedge]
            src_z = z[src_atom]
            tar_z = z[tar_atom]
            
            src_basis = basis_def[src_z]
            tar_basis = basis_def[tar_z]
            
            # Calculate orbital offsets for source and target atoms
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            # Reshape hamiltonian sub-matrix
            H_sub = hamiltonian[len(z) + iedge].reshape(nao_max, nao_max)
          
            
            # Fill matrix elements and convert to eV
            for i in range(len(src_basis)):
                for j in range(len(tar_basis)):
                    R_dict[R_vec][src_offset + i, tar_offset + j] = H_sub[src_basis[i], tar_basis[j]] * au2ev
    
    else:  # SOC case - complex matrices
        Hon = graph.Hon.numpy()
        Hoff = graph.Hoff.numpy()
        iHon = graph.iHon.numpy()
        iHoff = graph.iHoff.numpy()
        
        # For SOC, the matrix dimension is doubled (spin up/down)
        total_orbs = 2 * sum(len(basis_def[z[i]]) for i in range(len(z)))
        nao_max = int(np.sqrt(Hon.shape[1]) // 2)
        print(f'Total orbitals (with SOC): {total_orbs}, NAO max: {nao_max}')
        
        # Initialize R = (0,0,0) matrix for on-site terms
        R_dict[(0, 0, 0)] = np.zeros((total_orbs, total_orbs), dtype=complex)
        
        # Fill on-site terms (complex)
        orbital_offset = 0
        for iatm in range(len(z)):
            src_z = z[iatm]
            src_basis = basis_def[src_z]
            n_orb = len(src_basis)
            
            # Reconstruct complex matrix from real and imaginary parts
            H_real = Hon[iatm].reshape(2*nao_max, 2*nao_max)
            H_imag = iHon[iatm].reshape(2*nao_max, 2*nao_max)
            H_complex = (H_real + 1j * H_imag) * au2ev  # Convert to eV
            
            # Fill relevant blocks
            for spin1 in range(2):  # spin indices
                for spin2 in range(2):
                    for i in range(n_orb):
                        for j in range(n_orb):
                            row_idx = orbital_offset + spin1 * (total_orbs // 2) + i
                            col_idx = orbital_offset + spin2 * (total_orbs // 2) + j
                            orbital_i = src_basis[i] + spin1 * nao_max
                            orbital_j = src_basis[j] + spin2 * nao_max
                            R_dict[(0, 0, 0)][row_idx, col_idx] = H_complex[orbital_i, orbital_j]

            orbital_offset += n_orb
        
        # Fill off-site terms (similar logic for complex case)
        for iedge in range(edge_index.shape[1]):
            R_vec = tuple(cell_shift[iedge])
            if R_vec not in R_dict:
                R_dict[R_vec] = np.zeros((total_orbs, total_orbs), dtype=complex)
            
            src_atom = edge_index[0, iedge]
            tar_atom = edge_index[1, iedge]
            src_z = z[src_atom]
            tar_z = z[tar_atom]
            
            src_basis = basis_def[src_z]
            tar_basis = basis_def[tar_z]
            
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            H_real = Hoff[iedge].reshape(2*nao_max, 2*nao_max)
            H_imag = iHoff[iedge].reshape(2*nao_max, 2*nao_max)
            H_complex = (H_real + 1j * H_imag) * au2ev  # Convert to eV
            
            for spin1 in range(2):
                for spin2 in range(2):
                    for i in range(len(src_basis)):
                        for j in range(len(tar_basis)):
                            row_idx = src_offset + spin1 * (total_orbs // 2) + i
                            col_idx = tar_offset + spin2 * (total_orbs // 2) + j
                            orbital_i = src_basis[i] + spin1 * nao_max
                            orbital_j = tar_basis[j] + spin2 * nao_max
                            R_dict[R_vec][row_idx, col_idx] = H_complex[orbital_i, orbital_j]


    ######
    # Write to CSR file
    with open(output_file, 'w') as f:
        f.write(f"Matrix Dimension of H(R): {total_orbs}\n")
        f.write(f"Matrix number of H(R): {len(R_dict)}\n")
        
        for R_vec, H_mat in R_dict.items():
            # Convert to CSR format
            if H_mat.dtype == complex:
                # For complex matrices, write real and imaginary parts
                H_sparse = sp.csr_matrix(H_mat)
                nnz = H_sparse.nnz
                
                f.write(f"{R_vec[0]} {R_vec[1]} {R_vec[2]} {nnz}\n")
                
                # Write complex data as (real, imag) pairs
                complex_data = []
                for val in H_sparse.data:
                    complex_data.extend([val.real, val.imag])
                f.write(" ".join(map(str, complex_data)) + "\n")
                f.write(" ".join(map(str, H_sparse.indices)) + "\n")
                f.write(" ".join(map(str, H_sparse.indptr)) + "\n")
            else:
                # For real matrices
                H_sparse = sp.csr_matrix(H_mat)
                nnz = H_sparse.nnz
                
                f.write(f"{R_vec[0]} {R_vec[1]} {R_vec[2]} {nnz}\n")
                if nnz>0:
                    f.write(" ".join(map(str, H_sparse.data)) + "\n")
                    f.write(" ".join(map(str, H_sparse.indices)) + "\n")
                    f.write(" ".join(map(str, H_sparse.indptr)) + "\n")
        
    print(f'Hamiltonian CSR data saved to {output_file} (units: eV)')

def gen_S_csr(graph, output_file,nao_max=26,soc_switch=False):
    """
    Convert graph overlap matrix data to CSR format file
    
    Args:
        graph: graph data containing overlap matrix information
        output_file: path to output CSR file
    """
    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    elif nao_max == 26:
        basis_def = basis_def_26
    else:
        raise NotImplementedError
    
    # Extract data from graph
    z = graph.z.numpy()
    pos = graph.pos.numpy()
    edge_index = graph.edge_index.numpy()
    cell_shift = graph.cell_shift.numpy()
    overlap = graph.overlap.numpy()
    
    # Calculate total number of orbitals
    total_orbs = sum(len(basis_def[z[i]]) for i in range(len(z)))
    
    # Dictionary to store S(R) matrices
    R_dict = {}
    
    
    # Initialize R = (0,0,0) matrix for on-site terms
    R_dict[(0, 0, 0)] = np.zeros((total_orbs, total_orbs))
    
    # Fill on-site terms
    orbital_offset = 0
    for iatm in range(len(z)):
        src_z = z[iatm]
        src_basis = basis_def[src_z]
        n_orb = len(src_basis)
        
        if soc_switch:
            # For SOC case, overlap matrix structure is block diagonal in spin space
            # S = [[S_orbital,    0     ],
            #      [    0,    S_orbital]]
            S_sub = overlap[iatm].reshape(2*nao_max, 2*nao_max)
            
            # Fill spin blocks
            for spin1 in range(2):
                for spin2 in range(2):
                    for i in range(n_orb):
                        for j in range(n_orb):
                            row_idx = orbital_offset + spin1 * (total_orbs // 2) + i
                            col_idx = orbital_offset + spin2 * (total_orbs // 2) + j
                            orbital_i = src_basis[i] + spin1 * nao_max
                            orbital_j = src_basis[j] + spin2 * nao_max
                            
                            # Overlap matrix is block diagonal in spin space
                            if spin1 == spin2:
                                R_dict[(0, 0, 0)][row_idx, col_idx] = S_sub[orbital_i, orbital_j]
        else:
            # Non-SOC case
            S_sub = overlap[iatm].reshape(nao_max, nao_max)
            
            # Extract relevant elements based on basis
            for i in range(n_orb):
                for j in range(n_orb):
                    R_dict[(0, 0, 0)][orbital_offset + i, orbital_offset + j] = S_sub[src_basis[i], src_basis[j]]
        
        orbital_offset += n_orb

    
    # Fill off-site terms
    for iedge in range(edge_index.shape[1]):
        R_vec = tuple(cell_shift[iedge])
        if R_vec not in R_dict:
            R_dict[R_vec] = np.zeros((total_orbs, total_orbs))
        
        src_atom = edge_index[0, iedge]
        tar_atom = edge_index[1, iedge]
        src_z = z[src_atom]
        tar_z = z[tar_atom]
        
        src_basis = basis_def[src_z]
        tar_basis = basis_def[tar_z]
        
        if soc_switch:
            # Calculate orbital offsets for SOC case
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            S_sub = overlap[len(z) + iedge].reshape(2*nao_max, 2*nao_max)
            
            # Fill spin blocks for off-site terms
            for spin1 in range(2):
                for spin2 in range(2):
                    for i in range(len(src_basis)):
                        for j in range(len(tar_basis)):
                            row_idx = src_offset + spin1 * (total_orbs // 2) + i
                            col_idx = tar_offset + spin2 * (total_orbs // 2) + j
                            orbital_i = src_basis[i] + spin1 * nao_max
                            orbital_j = tar_basis[j] + spin2 * nao_max
                            
                            # Overlap matrix is block diagonal in spin space
                            if spin1 == spin2:
                                R_dict[R_vec][row_idx, col_idx] = S_sub[orbital_i, orbital_j]
        else:
            # Non-SOC case
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            S_sub = overlap[len(z) + iedge].reshape(nao_max, nao_max)
            
            # Fill matrix elements
            for i in range(len(src_basis)):
                for j in range(len(tar_basis)):
                    R_dict[R_vec][src_offset + i, tar_offset + j] = S_sub[src_basis[i], tar_basis[j]]
    
    # Write to CSR file
    with open(output_file, 'w') as f:
        f.write(f"Matrix Dimension of S(R): {total_orbs}\n")
        f.write(f"Matrix number of S(R): {len(R_dict)}\n")
        
        for R_vec, S_mat in R_dict.items():
            # Convert to CSR format (overlap matrix is always real)
            S_sparse = sp.csr_matrix(S_mat.real)  # Ensure real values
            nnz = S_sparse.nnz
            
            f.write(f"{R_vec[0]} {R_vec[1]} {R_vec[2]} {nnz}\n")
            if nnz>0:
                f.write(" ".join(map(str, S_sparse.data)) + "\n")
                f.write(" ".join(map(str, S_sparse.indices)) + "\n")
                f.write(" ".join(map(str, S_sparse.indptr)) + "\n")
    
    print(f'Overlap matrix CSR data saved to {output_file}')
def gen_R_csr(graph, output_file, nao_max=26, soc_switch=False):
    """
    Convert graph R matrix data to CSR format file
    
    Args:
        graph: graph data containing R matrix information  
        output_file: path to output CSR file
        nao_max: maximum number of atomic orbitals
        soc_switch: whether SOC is enabled
    """
    
    # Check if R matrix data exists
    if not hasattr(graph, 'R'):
        raise ValueError("Graph does not contain R matrix data!")
    
    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    elif nao_max == 26:
        basis_def = basis_def_26
    else:
        raise NotImplementedError
    
    # Extract data from graph
    z = graph.z.numpy()
    pos = graph.pos.numpy()
    edge_index = graph.edge_index.numpy()
    cell_shift = graph.cell_shift.numpy()
    R = graph.R.numpy()  # Shape: (3, num_sub_matrix, nao_max**2)
    
    # Calculate total number of orbitals
    is_soc = soc_switch
    if is_soc:
        # For SOC case, each orbital becomes 2 (spin up + spin down)
        total_orbs = 2 * sum(len(basis_def[z[i]]) for i in range(len(z)))
        # Keep original nao_max for matrix reshaping
        original_nao_max = nao_max
        nao_max_per_spin = nao_max // 2 if nao_max > 14 else nao_max
    else:
        total_orbs = sum(len(basis_def[z[i]]) for i in range(len(z)))
        original_nao_max = nao_max
        nao_max_per_spin = nao_max
    
    # Dictionary to store R(R) matrices for each direction
    # Structure: R_dict[R_vec] = [Rx_matrix, Ry_matrix, Rz_matrix]
    R_dict = {}
    
    # Initialize R = (0,0,0) matrices for on-site terms
    R_dict[(0, 0, 0)] = [np.zeros((total_orbs, total_orbs)) for _ in range(3)]
    
    # Fill on-site terms
    orbital_offset = 0
    for iatm in range(len(z)):

        src_z = z[iatm]
        src_basis = basis_def[src_z]
        n_orb = len(src_basis)
        
        if is_soc:
            # For SOC case, R matrix structure
            for direction in range(3):  # x, y, z directions
                R_sub = R[direction, iatm].reshape(original_nao_max, original_nao_max)
                
                # Fill both spin blocks (assuming R matrix is spin-independent)
                for spin in range(2):
                    for i in range(n_orb):
                        for j in range(n_orb):
                            row_idx = orbital_offset + spin * (total_orbs // 2) + i
                            col_idx = orbital_offset + spin * (total_orbs // 2) + j
                            R_dict[(0, 0, 0)][direction][row_idx, col_idx] = R_sub[src_basis[i], src_basis[j]]
        else:
            # Non-SOC case
            for direction in range(3):  # x, y, z directions
                R_sub = R[direction, iatm].reshape(original_nao_max, original_nao_max)

                
                # Extract relevant elements based on basis
                for i in range(n_orb):
                    for j in range(n_orb):
                        R_dict[(0, 0, 0)][direction][orbital_offset + i, orbital_offset + j] = R_sub[src_basis[i], src_basis[j]]

        orbital_offset += n_orb

    # Fill off-site terms
    for iedge in range(edge_index.shape[1]):
        R_vec = tuple(cell_shift[iedge])
        if R_vec not in R_dict:
            R_dict[R_vec] = [np.zeros((total_orbs, total_orbs)) for _ in range(3)]
        
        src_atom = edge_index[0, iedge]

        tar_atom = edge_index[1, iedge]
        src_z = z[src_atom]
        tar_z = z[tar_atom]
        src_basis = basis_def[src_z]
        tar_basis = basis_def[tar_z]
        
        if is_soc:
            # Calculate orbital offsets for SOC case
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            for direction in range(3):  # x, y, z directions
                R_sub = R[direction, len(z) + iedge].reshape(original_nao_max, original_nao_max)
                
                # Fill both spin blocks
                for spin in range(2):
                    for i in range(len(src_basis)):
                        for j in range(len(tar_basis)):
                            row_idx = src_offset + spin * (total_orbs // 2) + i
                            col_idx = tar_offset + spin * (total_orbs // 2) + j
                            R_dict[R_vec][direction][row_idx, col_idx] = R_sub[src_basis[i], tar_basis[j]]
        else:
            # Non-SOC case
            src_offset = sum(len(basis_def[z[i]]) for i in range(src_atom))
            tar_offset = sum(len(basis_def[z[i]]) for i in range(tar_atom))
            
            for direction in range(3):  # x, y, z directions
                R_sub = R[direction, len(z) + iedge].reshape(original_nao_max, original_nao_max)
                
                # Fill matrix elements
                for i in range(len(src_basis)):
                    for j in range(len(tar_basis)):
                        R_dict[R_vec][direction][src_offset + i, tar_offset + j] = R_sub[src_basis[i], tar_basis[j]]
    
    # Write to CSR file
    with open(output_file, 'w') as f:
        f.write(f"Matrix Dimension of r(R): {total_orbs}\n")
        f.write(f"Matrix number of r(R): {len(R_dict)}\n")
        
        for R_vec, R_matrices in R_dict.items():
            # Write R vector coordinates
            f.write(f"{R_vec[0]} {R_vec[1]} {R_vec[2]}\n")
            
            # Write three matrices: Rx, Ry, Rz
            for direction, R_mat in enumerate(R_matrices):
                # Convert to CSR format
                R_sparse = sp.csr_matrix(R_mat.real)  # Ensure real values
                nnz = R_sparse.nnz
                
                # Write number of non-zero elements
                f.write(f"{nnz}\n")
                
                # Write CSR data
                # 根据 pyatb 源码，只有 nnz > 0 时才写入 data, indices, 和 indptr
                if nnz > 0:
                    f.write(" ".join(map(str, R_sparse.data)) + "\n")
                    f.write(" ".join(map(str, R_sparse.indices)) + "\n")
                    f.write(" ".join(map(str, R_sparse.indptr)) + "\n")
    
    print(f'R matrix CSR data saved to {output_file}')

def gen_csr(graph,output_path,set_hamiltonian=None):
    output_H = os.path.join(output_path, "H.csr")
    output_S = os.path.join(output_path, "S.csr")
    output_R = os.path.join(output_path, "R.csr")
    gen_H_csr(graph, output_H, set_hamiltonian)
    gen_S_csr(graph, output_S)
    gen_R_csr(graph, output_R)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSR data from graph hamiltonian")
    parser.add_argument('--graph_data_path', type=str, required=True, help='Path to input graph data file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output CSR file')
    parser.add_argument('--set_hamiltonian_path', type=str, default=None, help='Path to set hamiltonian file (optional)')
    
    args = parser.parse_args()
    graph_data_path = args.graph_data_path
    with np.load(graph_data_path, allow_pickle=True) as npz:
    # 假设数据总是存在'graph'这个键下
            if 'graph' not in npz:
                raise KeyError(f"在 {graph_data_path} 中找不到必需的 'graph' 数据。")
            loaded_object = npz['graph'].item()
    list_of_graphs = list(loaded_object.values())
    graph=list_of_graphs[0]  # Assuming we only need the first graph for CSR generation
    set_hamiltonian_path = args.set_hamiltonian_path
    if set_hamiltonian_path is not None: 
        set_hamiltonian = np.load(set_hamiltonian_path) 
    else:
        set_hamiltonian = None

    gen_csr(graph, args.output_path, set_hamiltonian)