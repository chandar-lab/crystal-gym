from __future__ import annotations
from typing import List, Tuple, Union

import numpy as np
import torch
import dgl

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from crystal_gym.utils.utils import get_device


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

def build_crystal(crystal_str: str, niggli: bool = True, primitive: bool = False) -> Structure:
    """Build crystal from cif string.

    Args:
        crystal_str: CIF string representation of crystal
        niggli: Whether to apply Niggli reduction
        primitive: Whether to get primitive structure

    Returns:
        Structure object
    """
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal

def build_crystal_graph(
    crystal: Structure,
    species_ind: dict,
    graph_method: str = 'crystalnn',
    vocab_size: int = 88,
    substitution: bool = False,
) -> dgl.DGLGraph:
    """Build crystal graph from structure.

    Args:
        crystal: Crystal structure
        species_ind: Species index mapping
        graph_method: Method for building graph
        vocab_size: Vocabulary size for atomic numbers
        substitution: Whether to use random substitution

    Returns:
        DGL graph representation
    """

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    true_atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    num_atoms = len(true_atom_types)
    coords = frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    true_atom_types = np.array(true_atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)

    device = get_device()
    g = dgl.DGLGraph()
    g.add_nodes(num_atoms)
    edge_indices = torch.tensor(np.array(edge_indices))
    
    if substitution:
        g.ndata['atomic_number'] = torch.tensor(np.random.choice(vocab_size, num_atoms))
    else:
        g.ndata['atomic_number'] = torch.ones((num_atoms)) * vocab_size
    
    g.ndata['true_atomic_number'] = torch.tensor(true_atom_types)
    g.ndata['coords'] = torch.tensor(coords)
    g.add_edges(edge_indices[:, 0], edge_indices[:, 1])
    g.edata['to_jimages'] = torch.tensor(to_jimages)
    g.lengths = torch.tensor(lengths)
    g.angles = torch.tensor(angles)
    
    return g.to(device=device) 

def frac_to_cart_coords(
    frac_coords: np.ndarray,
    lengths: List[float],
    angles: List[float],
    num_atoms: int,
) -> torch.Tensor:
    lattice = lattice_params_to_matrix(lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2])
    lattice_nodes = torch.repeat_interleave(torch.tensor(lattice).reshape((1, 3, 3)), num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', torch.tensor(frac_coords), lattice_nodes)

    return pos

def lattice_params_to_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert lattice parameters to matrix.

    Args:
        a, b, c: Lattice lengths
        alpha, beta, gamma: Lattice angles in degrees

    Returns:
        Lattice matrix
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def abs_cap(val, max_abs_val=1):
    """
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val: Input value
        max_abs_val: Maximum absolute value (default: 1)

    Returns:
        Capped value
    """
    return max(min(val, max_abs_val), -max_abs_val)
