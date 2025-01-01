import numpy as np
import torch
import dgl
from crystal_gym.utils import get_pbc_distances
from crystal_gym.utils.variables import SPECIES_IND_SMALL_INV

def collate_function_crysrl(batch, p_hat, device = 'cuda'):

    edges_u, edges_v = batch.edges()
    edges_cat = torch.cat((edges_u[:, None], edges_v[:,None]), dim = 1)
    to_jimages = batch.edata['to_jimages']
    positions = batch.ndata['coords'].to(dtype = torch.float32)
    la = torch.cat((batch.lengths, batch.angles)).to(dtype = torch.float32)
    # batch.ndata['atomic_number'][batch.ndata['atomic_number'] == 88.0] = 18.0
    num_edges = edges_cat.shape[0]
    n_atoms = batch.ndata['atomic_number'].shape[0]
    out = get_pbc_distances(positions, edges_cat, lengths = la[None, :3], angles = la[None, 3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms, num_bonds=num_edges, coord_is_cart=True)
    

    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = n_atoms)
    g.ndata['atomic_number'] = batch.ndata['atomic_number']
    g.ndata['position'] = positions
    g.edata['e_feat'] = out['distances']
    g.edata['etype'] = to_jimages
    g.to(device = device)
    la = torch.cat((la, torch.tensor([p_hat]).to(dtype = torch.float32)))[None, :]
    g.lengths_angles_focus = la.to(device = device)
    g.focus = torch.tensor([batch.focus]).to(device = device)    
    g.focus_list = batch.focus_list
    
    return g