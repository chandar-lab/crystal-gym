from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import torch
import dgl
import numpy as np
from pymatgen.core.structure import Structure


def get_device() -> torch.device:
    """Get the appropriate device for tensor operations."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lattice_params_to_matrix_torch(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Batched torch version to compute lattice matrix from params.

    Args:
        lengths: torch.Tensor of shape (N, 3), unit A
        angles: torch.Tensor of shape (N, 3), unit degree
    
    Returns:
        Lattice matrix tensor
    """
    # Convert angles to radians and calculate trig functions
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    # Calculate gamma_star (derived angle) with numerical stability
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    # Construct lattice vectors a, b, c in Cartesian coordinates
    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def get_pbc_distances(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    to_jimages: torch.Tensor,
    num_atoms: List[int],
    num_bonds: List[int],
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute periodic boundary condition distances.
    
    This function calculates distances between atoms considering periodic boundary conditions,
    which is crucial for crystal structure analysis.
    
    Args:
        coords: Atomic coordinates (fractional or Cartesian)
        edge_index: Edge indices defining atom pairs
        lengths: Lattice lengths (a, b, c)
        angles: Lattice angles (alpha, beta, gamma)
        to_jimages: Periodic image offsets for each edge
        num_atoms: Number of atoms per structure in batch
        num_bonds: Number of bonds per structure in batch
        coord_is_cart: Whether coordinates are in Cartesian format
        return_offsets: Whether to return periodic offsets
        return_distance_vec: Whether to return distance vectors
    
    Returns:
        Dictionary containing edge indices, distances, and optionally offsets/vectors
    """
    # Convert lattice parameters to matrix for coordinate transformations
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Handle coordinate system conversion
    if coord_is_cart:
        pos = coords
    else:
        # Convert fractional to Cartesian coordinates
        lattice_nodes = torch.repeat_interleave(lattice, torch.tensor(num_atoms, device=lattice.device), dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)

    # Calculate distance vectors between connected atoms
    j_index, i_index = edge_index[:,0], edge_index[:,1]
    distance_vectors = pos[j_index] - pos[i_index]

    # Apply periodic boundary conditions
    lattice_edges = torch.repeat_interleave(lattice, torch.tensor(num_bonds, device=lattice.device), dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

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


def collate_function(batch: List, agent: str) -> Tuple[Union[dgl.DGLGraph, List], Union[dgl.DGLGraph, List], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching data from different RL agents.
    
    This function processes batches of experience data and converts them into
    the appropriate format for different reinforcement learning agents.
    
    Args:
        batch: List of experience tuples (observation, next_observation, action, reward, done, infos)
        agent: Type of agent ("MEGNetRL" or "CHGNetRL")
    
    Returns:
        Tuple containing:
        - Current state graphs/structures
        - Next state graphs/structures  
        - Actions taken
        - Rewards received
        - Done flags
    """
    batch_size = len(batch)
    
    # Handle MEGNetRL agent - uses DGL graphs for crystal structures
    if agent == "MEGNetRL":
        # Initialize data storage lists for batch processing
        atomic_number_list = [None] * batch_size
        position_list = [None] * batch_size
        laf_list = [None] * batch_size
        edges_u = [None] * batch_size
        edges_v = [None] * batch_size
        to_jimages = [None] * batch_size
        sum_n_atoms = 0
        n_atoms_list = []
        n_edges_list = []
        action_list = []
        atomic_number_list_next = [None] * batch_size
        num_edges = []
        reward_list = []
        bandgap_list = []
        dones_list = []
        focus_list = []
        focus_list_next = []
        efeat_list = []

        device = get_device()
        
        # Process each sample in the batch
        for i in range(batch_size):
            data_dict = batch[i]
            observation, next_observation, action, reward, done, infos = data_dict
            
            # Extract atomic numbers (last element is focus atom)
            atomic_number_list[i] = observation['atomic_number'][:-1]
            focus_list.append(observation['atomic_number'][-1])
            n_atoms = observation['atomic_number'].shape[0] - 1
            
            position_list[i] = observation['coordinates'].to(device=device)
            laf_list[i] = observation['laf']
            
            # Process edges with global node indexing across batch
            edges_u_single = observation['edges'][0] + sum_n_atoms
            edges_v_single = observation['edges'][1] + sum_n_atoms
            edges_cat = torch.cat([edges_u_single[:, None], edges_v_single[:, None]], dim=1).to(device=device)
            num_edges.append(edges_cat.shape[0])
            
            to_jimages[i] = observation['etype'].cpu()
            edges_u[i] = edges_cat[:, 0]
            edges_v[i] = edges_cat[:, 1]
            n_edges_list.append(edges_cat.shape[0])
            
            sum_n_atoms += n_atoms
            n_atoms_list.append(n_atoms)

            action_list.append(action)
            reward_list.append(reward)
            dones_list.append(done)

            atomic_number_list_next[i] = next_observation['atomic_number'][:-1]
            focus_list_next.append(next_observation['atomic_number'][-1])
        
        # Combine all data across the batch
        edges_cat = torch.cat([torch.cat(edges_u)[:, None].to(device=device), torch.cat(edges_v)[:, None].to(device=device)], dim=1)
        position = torch.cat(position_list, dim=0)
        laf_list = torch.stack(laf_list)
        to_jimages = torch.cat(to_jimages, dim=0)
        
        # Calculate periodic boundary condition distances
        out = get_pbc_distances(position.cpu(), edges_cat.cpu(), lengths=laf_list[:, :3].cpu(), angles=laf_list[:, 3:6].cpu(), to_jimages=to_jimages.cpu(), 
                            num_atoms=n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
        edata = out['distances']

        # Create DGL graph for current state
        g = dgl.graph(data=torch.unbind(edges_cat, dim=1), num_nodes=sum_n_atoms)
        g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim=0)
        g.focus = torch.stack(focus_list).to(device=device, dtype=torch.int64)
        g.lengths_angles_focus = laf_list.to(device=device)
        g.edata['e_feat'] = edata.to(device=device)
        g.edata['etype'] = to_jimages.to(device=device)
        g.set_batch_num_nodes(torch.tensor(n_atoms_list).to(device=device))
        g.set_batch_num_edges(torch.tensor(n_edges_list).to(device=device))

        # Create DGL graph for next state
        g_next = dgl.graph(data=torch.unbind(edges_cat, dim=1), num_nodes=sum_n_atoms)
        g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim=0)
        g_next.focus = torch.stack(focus_list_next).to(device=device, dtype=torch.int64)
        g_next.ndata['position'] = torch.cat(position_list, dim=0)
        g_next.lengths_angles_focus = laf_list.to(device=device)
        g_next.edata['e_feat'] = edata.to(device=device)
        g_next.edata['etype'] = to_jimages.to(device=device)
        g_next.set_batch_num_nodes(torch.tensor(n_atoms_list).to(device=device))
        g_next.set_batch_num_edges(torch.tensor(n_edges_list).to(device=device))

        g = g.to(device=device)
        g_next = g_next.to(device=device)
        
        # Convert RL data to tensors
        action_list = torch.tensor(np.array(action_list)).to(device=device)
        reward_list = torch.tensor(np.array(reward_list)).to(device=device)
        dones_list = torch.tensor(np.array(dones_list)).to(device=device)

        return g, g_next, action_list, reward_list, dones_list
    
    # Handle CHGNetRL agent - uses pymatgen Structure objects
    elif agent == "CHGNetRL":
        obs_list = []
        next_obs_list = []
        action_list = []
        reward_list = []
        dones_list = []
        
        for i in range(batch_size):
            data_dict = batch[i]
            observation, next_observation, action, reward, done, infos = data_dict
            
            # Convert to pymatgen Structure objects
            obs_struct = Structure.from_dict(observation)
            next_obs_struct = Structure.from_dict(next_observation)
            
            obs_list.append(obs_struct)
            next_obs_list.append(next_obs_struct)
            action_list.append(action)
            reward_list.append(reward)
            dones_list.append(done)
            
        device = get_device()
        action_list = torch.tensor(np.array(action_list)).to(device=device)
        reward_list = torch.tensor(np.array(reward_list)).to(device=device)
        dones_list = torch.tensor(np.array(dones_list)).to(device=device)
        
        return obs_list, next_obs_list, action_list, reward_list, dones_list

def create_graph(batch, p_hat: float, device: Optional[str] = None) -> dgl.DGLGraph:
    """
    Create a DGL graph from batch data with focus atom information.
    
    This function converts batch data into a DGL graph representation suitable
    for MEGNetRL agent processing, including focus atom and lattice information.
    
    Args:
        batch: Batch data containing atomic and structural information
        p_hat: Focus atom probability or parameter
        device: Device to place tensors on (if None, auto-detect)
    
    Returns:
        dgl.DGLGraph: Graph with node features, edge features, and metadata
    """
    if device is None:
        device = get_device()
    
    # Extract edge and coordinate information
    edges_u, edges_v = batch.edges()
    edges_cat = torch.cat((edges_u[:, None], edges_v[:, None]), dim=1).to(device=device)
    
    to_jimages = batch.edata['to_jimages'].to(device=device)
    positions = batch.ndata['coords'].to(dtype=torch.float32).to(device=device)
    la = torch.cat((batch.lengths, batch.angles)).to(dtype=torch.float32).to(device=device)
    
    num_edges = edges_cat.shape[0]
    n_atoms = batch.ndata['atomic_number'].shape[0]
    
    # Calculate periodic boundary condition distances
    out = get_pbc_distances(positions, edges_cat, lengths=la[None, :3], angles=la[None, 3:6], to_jimages=to_jimages, 
                            num_atoms=[n_atoms], num_bonds=[num_edges], coord_is_cart=True)
    
    # Create DGL graph with node and edge features
    g = dgl.graph(data=torch.unbind(edges_cat, dim=1), num_nodes=n_atoms)
    g.ndata['atomic_number'] = batch.ndata['atomic_number'].to(device=device)
    g.ndata['position'] = positions
    g.edata['e_feat'] = out['distances']
    g.edata['etype'] = to_jimages
    g.to(device=device)
    
    # Add focus atom information
    la = torch.cat((la, torch.tensor([p_hat], device=device).to(dtype=torch.float32)))[None, :]
    g.lengths_angles_focus = la.to(device=device)
    g.focus = torch.tensor([batch.focus], device=device)
    g.focus_list = batch.focus_list
    
    return g


def cart_to_frac_coords(
    cart_coords: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    """
    Convert Cartesian coordinates to fractional coordinates.
    
    Fractional coordinates are normalized coordinates where each component
    is between 0 and 1, representing the position within the unit cell.
    
    Args:
        cart_coords: Cartesian coordinates of atoms
        lengths: Lattice lengths (a, b, c)
        angles: Lattice angles (alpha, beta, gamma)
        num_atoms: Number of atoms in the structure
    
    Returns:
        torch.Tensor: Fractional coordinates (values between 0 and 1)
    """
    # Convert lattice parameters to matrix and calculate inverse
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, torch.tensor([num_atoms], device=inv_lattice.device), dim=0)
    
    # Transform Cartesian to fractional coordinates
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)
