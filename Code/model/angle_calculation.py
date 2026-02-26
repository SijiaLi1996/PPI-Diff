import numpy as np
from Bio.PDB import ProteinSequence
from itertools import groupby
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence

def calculate_all_angles(source_struct):
    """
    计算所有需要的角度。

    Args:
        source_struct: 使用 struc.parse_pdb 解析的结构。

    Returns:
        一个字典，包含所有角度。
    """

    # 定义要计算的角度
    angles = ["phi", "psi", "omega", "tau", "N:CA:C", "CA:C:1N", "C:1N:1CA"]

    # 定义要计算的距离
    distances = ["0C:1N", "N:CA", "CA:C"]

    # 计算二面角和距离
    try:
        phi, psi, omega = dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
    except BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None  # Return None if there's a BadStructureError

    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    backbone_atoms = source_struct[filter_backbone(source_struct)]
    for a in non_dihedral_angles:
        if a == "tau" or a == "N:CA:C":
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r, r + 1, r + 2]), np.zeros((3, 1))]).T
        elif a == "CA:C:1N":
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 1, r + 2, r + 3]), np.zeros((3, 1))]).T
        elif a == "C:1N:1CA":
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 2, r + 3, r + 4]), np.zeros((3, 1))]).T
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        calc_angles[a] = index_angle(backbone_atoms, indices=idx.astype(int))

    for k, v in calc_angles.items():
        if not (np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi):
            logging.warning(f"Illegal values for {k} in {fname} -- skipping")
            return None

    for d in distances:
        if (d == "0C:1N") or (d == "C:1N"):
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 2, r + 3]), np.zeros((2, 1))]).T
        elif d == "N:CA":
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r, r + 1]), np.zeros((2, 1))]).T
            assert len(idx) == len(calc_angles["phi"])
        elif d == "CA:C":
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r + 1, r + 2]), np.zeros((2, 1))]).T
            assert len(idx) == len(calc_angles["phi"])
        else:
            raise ValueError(f"Unrecognized distance: {d}")
        calc_angles[d] = index_distance(backbone_atoms, indices=idx.astype(int))

    return calc_angles

# 你提供的代码中的其他函数（dihedral_backbone、index_angle、index_distance、filter_backbone、BadStructureError 等）
# 需要复制到这里