import torch
import dgl
import numpy as np


NUM_WORKERS = 16
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lattice_params_to_matrix_torch(lengths, angles):
    """
    Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

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

    j_index, i_index = edge_index[:,0], edge_index[:,1]

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, torch.tensor(num_bonds), dim=0)
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


def collate_function(batch, p_hat):
    
    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
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

    for i in range(batch_size):
        data_dict = batch[i]
        observation, action, next_observation, reward, bandgap, done = data_dict
    
        atomic_number_list[i] = observation['atomic_number'][:-1]
        focus_list.append(observation['atomic_number'][-1])
        n_atoms = observation['atomic_number'].shape[0] - 1
        true_atomic_number_list[i] = observation['true_atomic_number']
        position_list[i] = observation['coordinates']
        if observation['laf'].shape[0] == 7:
            laf_list[i] = torch.cat((observation['laf'], torch.tensor([p_hat]))) 
        else:
            laf_list[i] = observation['laf']
        edges_u_single =  observation['edges'][0] + sum_n_atoms
        edges_v_single =  observation['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        # try:
        to_jimages[i] = observation['etype']
        # except:
            # efeat/_list.append(observation['efeat'])
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)

        action_list.append(action)
        reward_list.append(reward)
        bandgap_list.append(bandgap)
        dones_list.append(done)

        atomic_number_list_next[i] = next_observation['atomic_number'][:-1]
        focus_list_next.append(next_observation['atomic_number'][-1])
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)

    # try:
    to_jimages = torch.cat(to_jimages, dim = 0)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                        num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    edata = out['distances'] 
    # except:
        # edata = torch.cat(efeat_list)

    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.focus = torch.stack(focus_list)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.lengths_angles_focus = laf_list
    g.edata['e_feat'] = edata
    g.edata['etype'] = to_jimages
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.focus = torch.stack(focus_list_next)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = laf_list
    g_next.edata['e_feat'] = edata
    g_next.edata['etype'] = to_jimages
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))

    return g, torch.tensor(action_list), g_next, torch.tensor(reward_list), torch.tensor(bandgap_list), torch.tensor(dones_list)


def collate_function_eval(batch, p_hat):

    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    action_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size
    num_edges = []
    reward_list = []
    bandgap_list = []
    dones_list = []
    focus_list = []
    focus_list_next = []
    focus_all_list = []

    for i in range(batch_size):
        data_dict = batch[i]

        observation, action, next_observation, reward, done = data_dict
        atomic_number_list[i] = observation['atomic_number'][:-1]
        focus_list.append(observation['atomic_number'][-1])
        n_atoms = observation['atomic_number'].shape[0] - 1
        true_atomic_number_list[i] = observation['true_atomic_number']
        position_list[i] = observation['coordinates']
        laf_list[i] = torch.cat((observation['laf'], torch.tensor([p_hat]))) 
        edges_u_single =  observation['edges'][0] + sum_n_atoms
        edges_v_single =  observation['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        to_jimages[i] = observation['etype']
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        ind_list.append(observation['ind'])

        action_list.append(action)
        reward_list.append(reward)
        dones_list.append(done)

        atomic_number_list_next[i] = next_observation['atomic_number'][:-1]
        focus_list_next.append(next_observation['atomic_number'][-1])
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    edata = out['distances'] 
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = laf_list
    g.edata['e_feat'] = edata
    g.edata['etype'] = to_jimages
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    g.focus = torch.stack(focus_list)
    g.inds = torch.tensor(ind_list)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.focus = torch.stack(focus_list_next)
    g_next.lengths_angles_focus = laf_list
    g_next.edata['e_feat'] = edata
    g_next.edata['etype'] = to_jimages
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    g_next.inds = torch.tensor(ind_list)

    return g


def collate_function_crysrl(batch, p_hat):

    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    action_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size
    num_edges = []
    reward_list = []
    bandgap_list = []
    dones_list = []
    focus_list = []
    focus_list_next = []
    focus_all_list = []

    for i in range(batch_size):
        data_dict = batch[i]

        observation, action, next_observation, reward, done, focus = data_dict
        focus_all_list.append(focus)
        atomic_number_list[i] = observation['atomic_number'][:-1]
        focus_list.append(observation['atomic_number'][-1])
        n_atoms = observation['atomic_number'].shape[0] - 1
        true_atomic_number_list[i] = observation['true_atomic_number']
        position_list[i] = observation['coordinates']
        laf_list[i] = torch.cat((observation['laf'], torch.tensor([p_hat]))) 
        edges_u_single =  observation['edges'][0] + sum_n_atoms
        edges_v_single =  observation['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        to_jimages[i] = observation['etype']
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        ind_list.append(observation['ind'])

        action_list.append(action)
        reward_list.append(reward)
        dones_list.append(done)

        atomic_number_list_next[i] = next_observation['atomic_number'][:-1]
        focus_list_next.append(next_observation['atomic_number'][-1])
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    
    
    edata = out['distances'] 
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = laf_list
    g.edata['e_feat'] = edata
    g.edata['etype'] = to_jimages
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    g.focus = torch.stack(focus_list)
    g.inds = torch.tensor(ind_list)
    g.focus_all = focus_all_list[0]

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.focus = torch.stack(focus_list_next)
    g_next.lengths_angles_focus = laf_list
    g_next.edata['e_feat'] = edata
    g_next.edata['etype'] = to_jimages
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    g_next.inds = torch.tensor(ind_list)
    g_next.focus_all = focus_all_list[0]

    return g


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    """
    Cartesan to fractional coordinates
    Source: https://github.com/txie-93/cdvae/tree/main/cdvae
    """
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)


def create_sublists(A, B):
    output = []
    start_index = 0

    for num_items in B:
        sublist = torch.tensor(A[start_index:start_index+num_items.numpy()])
        output.append(sublist)
        start_index += num_items

    return output