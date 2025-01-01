import numpy as np
import torch
import copy
import itertools
import torch.nn.functional as F
import dgl

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from chgnet.model import CHGNet




CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
):
    """
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
    """
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
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

def build_crystal_graph(crystal, 
                        species_ind, 
                        graph_method='crystalnn',
                        substitution = False, 
                        vocab_size = 88):
    """
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
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
    # true_atom_types = torch.tensor([species_ind[i] for i in crystal.atomic_numbers])
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

    g = dgl.DGLGraph()#.to(device = 'cuda:0')
    g.add_nodes(num_atoms)
    edge_indices = torch.tensor(np.array(edge_indices))
    if substitution:
        g.ndata['atomic_number'] = torch.tensor(np.random.choice(vocab_size, num_atoms))
    else:
        g.ndata['atomic_number'] = torch.ones((num_atoms)) * vocab_size
    g.ndata['true_atomic_number'] = torch.tensor(true_atom_types) #.to(device = 'cuda:0')  ## 56 vocab size + 1 blank slot 
    g.ndata['coords'] = torch.tensor(coords)
    g.add_edges(edge_indices[:,0], edge_indices[:,1])
    g.edata['to_jimages'] = torch.tensor(to_jimages)
    g.lengths = torch.tensor(lengths)
    g.angles = torch.tensor(angles)

    return g 

def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix(lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2])
    lattice_nodes = torch.repeat_interleave(torch.tensor(lattice).reshape((1,3,3)), num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', torch.tensor(frac_coords), lattice_nodes)  # cart coords

    return pos

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
    Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
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
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)
