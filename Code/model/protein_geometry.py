import math
import os
import warnings
import numpy as np
from typing import List, Tuple, Optional

# Import necessary BioPython components
try:
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.Polypeptide import Polypeptide
    from Bio.PDB.vectors import calc_angle, calc_dihedral
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
except ImportError:
    print("Error: BioPython library not found.")
    print("Please install BioPython: pip install biopython")

# 类型别名
AngleTuple = Tuple[Optional[float], Optional[float], Optional[float],
                   Optional[float], Optional[float], Optional[float]]
AngleList = List[AngleTuple]


def _get_atom_vector(residue, atom_name: str) -> Optional[np.ndarray]:
    """安全地从残基中获取指定原子的坐标，返回为 NumPy 数组。"""
    try:
        # BioPython 2.0+ a .get_vector() method returning a Vector, older versions a numpy array
        # .get_coord() always returns a numpy array.
        return residue[atom_name].get_coord()
    except KeyError:
        return None


def calculate_backbone_angles(polypeptide: Polypeptide) -> AngleList:
    """健壮的函数，用于计算每个残基的六个主链角度。"""
    all_angles = []
    residue_triplets = zip([None] + polypeptide[:-1],
                           polypeptide,
                           polypeptide[1:] + [None])
    for prev_res, current_res, next_res in residue_triplets:
        phi, psi, omega, tau, ca_c_n, c_n_ca = None, None, None, None, None, None
        
        # 为了与 calc_dihedral/calc_angle 兼容，我们传递 Vector 对象
        # 但我们内部可以处理 numpy 数组
        def to_vector(coord_array):
            from Bio.PDB.vectors import Vector
            return Vector(coord_array) if coord_array is not None else None

        n_i, ca_i, c_i = to_vector(_get_atom_vector(current_res, "N")), to_vector(_get_atom_vector(current_res, "CA")), to_vector(_get_atom_vector(current_res, "C"))
        c_im1 = to_vector(_get_atom_vector(prev_res, "C")) if prev_res else None
        n_ip1 = to_vector(_get_atom_vector(next_res, "N")) if next_res else None
        ca_ip1 = to_vector(_get_atom_vector(next_res, "CA")) if next_res else None

        if n_i and ca_i and c_i: tau = calc_angle(n_i, ca_i, c_i)
        if c_im1 and n_i and ca_i and c_i: phi = calc_dihedral(c_im1, n_i, ca_i, c_i)
        if n_i and ca_i and c_i and n_ip1: psi = calc_dihedral(n_i, ca_i, c_i, n_ip1)
        if ca_i and c_i and n_ip1 and ca_ip1: omega = calc_dihedral(ca_i, c_i, n_ip1, ca_ip1)
        if ca_i and c_i and n_ip1: ca_c_n = calc_angle(ca_i, c_i, n_ip1)
        if c_i and n_ip1 and ca_ip1: c_n_ca = calc_angle(c_i, n_ip1, ca_ip1)
        
        all_angles.append((phi, psi, omega, tau, ca_c_n, c_n_ca))
    return all_angles


def process_angles_sin_cos(angles_list: AngleList) -> np.ndarray:
    """将角度元组列表转换为包含正弦和余弦值的 NumPy 数组。"""
    num_residues = len(angles_list)
    features = np.zeros((num_residues, 12))
    for i, angle_tuple in enumerate(angles_list):
        for j, angle_rad in enumerate(angle_tuple):
            if angle_rad is not None and not math.isnan(angle_rad):
                features[i, 2 * j] = math.sin(angle_rad)
                features[i, 2 * j + 1] = math.cos(angle_rad)
    return features


def _rigid_transform_3d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算将点集 A 对齐到点集 B 的旋转矩阵 R 和平移向量 t。"""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t


def _place_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond_length: float, bond_angle: float, torsion_angle: float) -> np.ndarray:
    """根据前三个原子 a, b, c 的位置，计算第四个原子 d 的坐标 (NeRF算法)。"""
    v_bc = c - b
    v_ab = b - a
    u_bc = v_bc / np.linalg.norm(v_bc)
    v_n = np.cross(v_ab, u_bc)
    u_n = v_n / np.linalg.norm(v_n)
    u_nbc = np.cross(u_n, u_bc)
    angle = np.pi - bond_angle
    d = c + (
        bond_length * np.sin(angle) * np.cos(torsion_angle) * u_nbc +
        bond_length * np.sin(angle) * np.sin(torsion_angle) * u_n -
        bond_length * np.cos(angle) * u_bc
    )
    return d


def angles_to_coordinates(angle_features_sin_cos: np.ndarray,
                          first_n_ca_c_coords: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 NeRF 算法将六个主链角度转换回3D坐标。"""
    num_residues = angle_features_sin_cos.shape[0]
    BOND_N_CA, BOND_CA_C, BOND_C_N = 1.458, 1.525, 1.329
    n_coords = np.full((num_residues, 3), np.nan)
    ca_coords = np.full((num_residues, 3), np.nan)
    c_coords = np.full((num_residues, 3), np.nan)
    
    angles = np.arctan2(angle_features_sin_cos[:, ::2], angle_features_sin_cos[:, 1::2])
    phi, psi, omega, tau, ca_c_n, c_n_ca = angles.T

    n_coords[0] = np.array([0.0, 0.0, 0.0])
    ca_coords[0] = np.array([BOND_N_CA, 0.0, 0.0])
    c_x = BOND_N_CA - BOND_CA_C * np.cos(np.pi - tau[0])
    c_y = BOND_CA_C * np.sin(np.pi - tau[0])
    c_coords[0] = np.array([c_x, c_y, 0.0])

    for i in range(1, num_residues):
        try:
            n_coords[i] = _place_atom(n_coords[i-1], ca_coords[i-1], c_coords[i-1], BOND_C_N, ca_c_n[i-1], psi[i-1])
            ca_coords[i] = _place_atom(ca_coords[i-1], c_coords[i-1], n_coords[i], BOND_N_CA, c_n_ca[i-1], omega[i-1])
            c_coords[i] = _place_atom(c_coords[i-1], n_coords[i], ca_coords[i], BOND_CA_C, tau[i], phi[i])
        except Exception:
            break

    if first_n_ca_c_coords is not None and not np.isnan(c_coords[0]).any():
        ref_atoms = np.array(first_n_ca_c_coords)
        mov_atoms = np.array([n_coords[0], ca_coords[0], c_coords[0]])
        
        R, t = _rigid_transform_3d(mov_atoms, ref_atoms)
        
        valid_mask = ~np.isnan(n_coords).any(axis=1)
        if np.any(valid_mask):
            n_coords[valid_mask] = (R @ n_coords[valid_mask].T).T + t
            ca_coords[valid_mask] = (R @ ca_coords[valid_mask].T).T + t
            c_coords[valid_mask] = (R @ c_coords[valid_mask].T).T + t
            
    return n_coords, ca_coords, c_coords

