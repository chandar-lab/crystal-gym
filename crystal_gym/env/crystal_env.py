import os
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import torch
import time
from pymatgen.io.ase import AseAtomsAdaptor
from crysrl.utils.create_graph import collate_function_crysrl
from crystal_design.utils import cart_to_frac_coords
from ase.calculators.espresso import Espresso, EspressoProfile
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core import Structure, Lattice
from crystal_design.utils.data_utils import build_crystal, build_crystal_graph
from crystal_design.utils.variables import ELEMENTS_SMALL, SPECIES_IND_INV, SPECIES_IND_SMALL
from dgl.traversal import bfs_nodes_generator
from crystal_gym.utils.variables import CUBIC_INDS_VAL

class CrystalGymEnv(gym.Env):
    def __init__(self, 
                 kwargs = {
                           'data_path':'../data/mp_20.csv',
                           'project':'project', 
                           'group':'group', 
                           'exp_name':'exp_name',
                           'seed':0,
                           'options':{'dataset':
                                    'mp_20',
                                    'mode':'single',
                                    'p_hat':1.12,
                                    'index':0,
                                    }}):
        
        """
        Initialize the CrystalGymEnv class.
        Args:
            data_path (str): The path to the data file.
            options (dict): A dictionary containing the project, group, and experiment
        """
        super(CrystalGymEnv, self).__init__()
        self.env_options = kwargs['env']
        self.run_name = kwargs['env']['run_name']
        ## Load the data
        self.data = pd.read_csv(self.env_options['data_path'])

        # Define the action and observation space
        self.action_space = self.single_action_space = gym.spaces.Discrete(len(ELEMENTS_SMALL))
        self.observation_space =  self.single_observation_space = gym.spaces.Box(low=0, high=100, shape=(1,)) # Dummy observation space; actual space is a graph

        ## DFT Inputs
        self.qe_inputs = kwargs['qe']
        self.pseudodict = pickle.load(open(kwargs['qe']['pseudodict'], 'rb'))
        pseudo_dir = kwargs['qe']['pseudo_dir']

        self.profile = EspressoProfile(
                command = "mpirun --bind-to none -np 1 /home/mila/p/prashant.govindarajan/scratch/qe-7.1/bin/pw.x",
                pseudo_dir = pseudo_dir,
            )

        # self.calculator = Espresso(profile = self.profile,
        #                             pseudopotentials=self.pseudodict,
        #                             input_data=self.qe_inputs, 
        #                             kpts=(4,4,4), 
        #                             directory= os.path.join('calculations', self.run_name))
        ####
        # self.project = options['project']    
        # self.group = options['group']
        # self.exp_name = options['exp_name']
        # self.env_options = kwargs
        self.state, _ = self.reset(self.env_options['seed'], {})
        self.t = 0

    def reset(self, 
            seed = 0, 
            options = {}):
        """
        Reset the environment.
        Returns:
            state (dict): The state of the environment.
        """
        info = {}
        if self.env_options['mode'] == 'single':
            self.sample_ind = self.env_options['index']
        elif self.env_options['mode'] == 'cubic':
            self.sample_ind = np.random.choice(CUBIC_INDS_VAL)
        
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        graph = build_crystal_graph(canonical_crystal, SPECIES_IND_INV)

        self.n_sites = graph.num_nodes()
        self.bfs_start = np.random.choice(self.n_sites) 
        self.err_flag = 0

        if self.bfs_start >= self.n_sites:
            self.err_flag = 1
            return None
        
        self.traversal = torch.cat(list(bfs_nodes_generator(graph, self.bfs_start)))

        try:
            assert len(self.traversal) == self.n_sites
        except:
            self.traversal = torch.tensor(list(range(self.n_sites)))
            self.err_flag = 1
        self.t = 0
    
        graph.focus = self.traversal[self.t]
        graph.focus_list = self.traversal
        state = collate_function_crysrl(graph, p_hat = self.env_options['p_hat'])

        lengths = torch.tensor(canonical_crystal.lattice.abc)
        angles = torch.tensor(canonical_crystal.lattice.angles)
        state.lengths_angles = torch.cat([lengths, angles])
        self.n_sites = state.num_nodes()
        self.err_flag = 0

        if self.bfs_start >= self.n_sites:
            self.err_flag = 1
            return None
        self.traversal = torch.cat(list(bfs_nodes_generator(state, self.bfs_start)))
        try:
            assert len(self.traversal) == self.n_sites
        except AssertionError:
            self.traversal = torch.tensor(list(range(self.n_sites)))
            self.err_flag = 1

        return state, info
    
    def compute_reward(self):
        """
        Compute the reward.
        Returns:
            reward (float): The reward.
        """
        error_flag = 0
        canonical_crystal = self.render()
        # species = set([sp.name for sp in canonical_crystal.species])
        atoms = AseAtomsAdaptor.get_atoms(canonical_crystal)
        nbnd = int(np.ceil(sum(atoms.get_atomic_numbers()) // 2 * 1.2))
        self.qe_inputs.update({'nbnd':nbnd})
        kpts = Kpoints.automatic_density(canonical_crystal, kppa = self.qe_inputs['kppa']).kpts[0]
        atoms.calc = Espresso(profile = self.profile,
                                pseudopotentials=self.pseudodict,
                                input_data=self.qe_inputs, 
                                kpts=kpts, 
                                directory= os.path.join('calculations', self.run_name))
        try:
            start_time = time.time()
            energy = atoms.get_potential_energy()
        except:
            pass
        try:
            end_time = time.time()        
            with open("/".join(['calculations/'+self.run_name, 'espresso.pwo']), 'r') as f:
                lines = f.read()
                if 'convergence NOT achieved after' in lines:
                    error_flag = 1
                    assert False
                elif 'charge is wrong' in lines:
                    error_flag = 2
                    assert False
                tmp = lines.split('highest occupied, lowest unoccupied level (ev):')[-1].split()[:2]
                bg = float(tmp[1]) - float(tmp[0])
                if bg < 0.0:
                    bg = 0.0
                reward = self.distance(self.env_options['p_hat'], torch.tensor([bg])).item()
                # print('DFT Success!')
                sim_time = end_time - start_time
                return reward, bg, error_flag, sim_time
        except:
            if error_flag == 0:
                error_flag = 3
            reward = -1.0

        return reward, None, error_flag, None
    
    def distance(self, target, predicted):
        """
        Compute the distance between two vectors.
        Args:
            x (torch.Tensor): The first vector.
            y (torch.Tensor): The second vector.
        Returns:
            distance (float): The distance between the two vectors.
        """
        try:
            d = torch.exp(-(target - predicted)**2 / 1.0)[0]
        except:
            d = torch.exp(-(target - predicted)**2 / 1.0)#[0]
        return d

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (int): The action to take.
        Returns:
            state (dict): The state of the environment.
            reward (float): The reward.
            terminated (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode is truncated.
            info (dict): Additional information
        """
        info = {}  
        atomic_number = deepcopy(self.state.ndata['atomic_number'])
        index_curr_focus = self.traversal[self.t]
        atomic_number[index_curr_focus] = torch.tensor(action)
        next_observations = deepcopy(self.state)
        next_observations.ndata['atomic_number'] = atomic_number
        self.state = deepcopy(next_observations)
        self.t += 1
        if self.t == self.n_sites:
            terminated = truncated = True
            self.t = 0
            reward, bg, error_flag, sim_time = self.compute_reward()
            info['final_info'] = [{'episode':{'r':reward}}]
            info['final_info'][0]['episode']['error_flag'] = error_flag
            if bg is not None:
                info['final_info'][0]['episode']['bg'] = bg
                info['final_info'][0]['episode']['sim_time'] = sim_time
        else:
            terminated = truncated = False
            reward = 0.0
        return self.state, reward, terminated, truncated, info
    
    def graph_to_dict_complete(self, observations):
        """
        Convert the graph to a dictionary.
        """
        atomic_number = deepcopy(observations.ndata['atomic_number'])
        position = deepcopy(observations.ndata['position'])
        lengths = deepcopy(observations.lengths_angles_focus.cpu()[0][:3])
        angles = deepcopy(observations.lengths_angles_focus.cpu()[0][3:6])
        num_atoms = atomic_number.shape[0]
        frac_coords = cart_to_frac_coords(position.to(dtype=torch.float32).cpu(), lengths.unsqueeze(0), angles.unsqueeze(0), num_atoms)
        state_dict = {
                      'frac_coords':np.array(frac_coords), 
                      'atom_types':np.array(atomic_number.cpu()), 
                      'lengths':np.array(lengths), 
                      'angles':np.array(angles), 
                      'num_atoms':num_atoms
                      }
        return state_dict
    
    def to_struct(self,state_dict):
        """
        Convert the dictionary to a pymatgen Structure.
        """
        lengths = state_dict['lengths'].tolist()
        angles = state_dict['angles'].tolist()
        lattice_params = lengths + angles
        atomic_number = state_dict['atom_types']
        atom_types = [SPECIES_IND_SMALL[int(atomic_number[j])] for j in range(atomic_number.shape[0])]
        coords = state_dict['frac_coords']
        canonical_crystal = Structure(lattice = Lattice.from_parameters(*lattice_params),
                                    species = atom_types, coords = coords)
        return canonical_crystal

    def render(self, mode="human"):
        """
        Render the environment.
        """
        crystal_dict = self.graph_to_dict_complete(self.state)
        canonical_crystal = self.to_struct(crystal_dict)
        return canonical_crystal
    
    def get_obs(self):
        """
        Get the observation.
        """
        return self.state
    
    def close(self):
        """
        Close the environment.
        """
        self.reset()
    