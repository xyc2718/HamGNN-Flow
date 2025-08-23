import json
import numpy as np
import scipy.sparse as sp
import argparse
from pathlib import Path
import math

# 定义单位转换常数
AU_TO_EV = 27.211386245988

def get_orbital_info(z, on_site_matrices):
    """
    从在位矩阵的尺寸动态推断每个原子的轨道数、总轨道数和偏移量。
    """
    num_orbs_per_atom = [int(math.sqrt(len(block))) for block in on_site_matrices]
    total_orbs = sum(num_orbs_per_atom)
    
    offsets = [0] * len(z)
    for i in range(1, len(z)):
        offsets[i] = offsets[i-1] + num_orbs_per_atom[i-1]
        
    return num_orbs_per_atom, total_orbs, offsets

def build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, on_site_data, off_site_data):
    """一个通用的函数，用于从紧凑的矩阵块构建全局矩阵字典。"""
    matrix_dict = {}

    # --- 处理在位项 ---
    for iatm, on_site_block_flat in enumerate(on_site_data):
        if (0, 0, 0) not in matrix_dict:
            matrix_dict[(0, 0, 0)] = np.zeros((total_orbs, total_orbs))
        
        n_orb = num_orbs_per_atom[iatm]
        offset = offsets[iatm]
        
        sub_matrix = np.array(on_site_block_flat).reshape(n_orb, n_orb)
        
        matrix_dict[(0, 0, 0)][offset:offset + n_orb, offset:offset + n_orb] = sub_matrix

    # --- 处理异位项 ---
    for iedge, off_site_block_flat in enumerate(off_site_data):
        r_vec = tuple(cell_shift[iedge])
        if r_vec not in matrix_dict:
            matrix_dict[r_vec] = np.zeros((total_orbs, total_orbs))
            
        src_atom, tar_atom = edge_index[0, iedge], edge_index[1, iedge]
        n_orb_src, n_orb_tar = num_orbs_per_atom[src_atom], num_orbs_per_atom[tar_atom]
        offset_src, offset_tar = offsets[src_atom], offsets[tar_atom]

        sub_matrix = np.array(off_site_block_flat).reshape(n_orb_src, n_orb_tar)
        
        matrix_dict[r_vec][offset_src:offset_src + n_orb_src, offset_tar:offset_tar + n_orb_tar] = sub_matrix
                                  
    return matrix_dict

def write_hs_csr(matrix_dict, total_orbs, output_file, matrix_name):
    """写入H.csr和S.csr文件的函数。"""
    print(f"Writing {matrix_name} matrix to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"Matrix Dimension of {matrix_name}(R): {total_orbs}\n")
        f.write(f"Matrix number of {matrix_name}(R): {len(matrix_dict)}\n")
        
        for r_vec, matrix in matrix_dict.items():
            sparse_mat = sp.csr_matrix(matrix)
            nnz = sparse_mat.nnz
            
            f.write(f"{r_vec[0]} {r_vec[1]} {r_vec[2]} {nnz}\n")
            if nnz > 0:
                f.write(" ".join(map(str, sparse_mat.data)) + "\n")
                f.write(" ".join(map(str, sparse_mat.indices)) + "\n")
                f.write(" ".join(map(str, sparse_mat.indptr)) + "\n")
    print("Done.")

def write_r_csr(r_matrices, total_orbs, output_file):
    """写入R.csr文件的函数，格式特殊。"""
    print(f"Writing R matrix to {output_file}...")
    rx_dict, ry_dict, rz_dict = r_matrices
    r_vectors = list(rx_dict.keys())
    
    with open(output_file, 'w') as f:
        f.write(f"Matrix Dimension of r(R): {total_orbs}\n")
        f.write(f"Matrix number of r(R): {len(r_vectors)}\n")

        for r_vec in sorted(r_vectors): # 排序以保证确定性
            f.write(f"{r_vec[0]} {r_vec[1]} {r_vec[2]}\n")
            
            for direction_matrix in [rx_dict[r_vec], ry_dict[r_vec], rz_dict[r_vec]]:
                sparse_mat = sp.csr_matrix(direction_matrix)
                nnz = sparse_mat.nnz
                
                f.write(f"{nnz}\n")
                if nnz > 0:
                    f.write(" ".join(map(str, sparse_mat.data)) + "\n")
                    f.write(" ".join(map(str, sparse_mat.indices)) + "\n")
                    f.write(" ".join(map(str, sparse_mat.indptr)) + "\n")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Generate PyATB CSR files directly from a self-contained HS.json file.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input HS.json file.")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Path to the output directory for CSR files.")
    args = parser.parse_args()

    # --- 1. 加载所有数据 ---
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # 从单一来源获取所有信息
    # z = np.array(data['z'])
    z = np.array([14,14])
    edge_index = np.array(data['edge_index'])
    cell_shift = np.array(data['cell_shift'])
    
    hon = data['Hon'][0]
    hoff = data['Hoff'][0]
    son = data['Son']
    soff = data['Soff']
    pon = data['Pon']
    poff = data['Poff']
    
    # --- 2. 动态获取轨道信息 ---
    num_orbs_per_atom, total_orbs, offsets = get_orbital_info(z, hon)
    print(f"Successfully deduced orbital info:")
    print(f" - Atoms: {len(z)}")
    print(f" - Orbitals per atom: {num_orbs_per_atom}")
    print(f" - Total orbitals: {total_orbs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 3. 生成 H.csr ---
    h_dict = build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, hon, hoff)
    for key in h_dict:
        h_dict[key] *= AU_TO_EV # 单位转换
    write_hs_csr(h_dict, total_orbs, args.output_dir / "H.csr", "H")
    
    # --- 4. 生成 S.csr ---
    s_dict = build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, son, soff)
    write_hs_csr(s_dict, total_orbs, args.output_dir / "S.csr", "S")

    # --- 5. 生成 R.csr ---
    # R 矩阵的构建逻辑不同，因为它有三个分量
    pon_x = [np.array(p)[:, 0] for p in pon]
    pon_y = [np.array(p)[:, 1] for p in pon]
    pon_z = [np.array(p)[:, 2] for p in pon]
    poff_x = [np.array(p)[:, 0] for p in poff]
    poff_y = [np.array(p)[:, 1] for p in poff]
    poff_z = [np.array(p)[:, 2] for p in poff]

    rx_dict = build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, pon_x, poff_x)
    ry_dict = build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, pon_y, poff_y)
    rz_dict = build_matrix_dict(num_orbs_per_atom, total_orbs, offsets, edge_index, cell_shift, pon_z, poff_z)
    write_r_csr([rx_dict, ry_dict, rz_dict], total_orbs, args.output_dir / "R.csr")
    
    print(f"\nAll CSR files have been generated successfully in: {args.output_dir}")

if __name__ == "__main__":
    main()