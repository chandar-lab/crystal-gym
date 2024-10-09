import os
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import torch
import ase
from pymatgen.io.ase import AseAtomsAdaptor
from crysrl.utils.create_graph import collate_function_crysrl
from crystal_design.utils import cart_to_frac_coords
from ase.calculators.espresso import Espresso, EspressoProfile
from pymatgen.core import Structure, Lattice
from crystal_design.utils.data_utils import build_crystal, build_crystal_graph
from crystal_design.utils.variables import ELEMENTS_SMALL, SPECIES_IND_INV, SPECIES_IND_SMALL
from dgl.traversal import bfs_nodes_generator

class CrystalGymEnv(gym.Env):
    def __init__(self, 
                 data_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crysrl/crysrl/data/mp_20/val.csv',
                 kwargs = {'project':'project', 
                           'group':'group', 
                           'exp_name':'exp_name',
                           'seed':0,
                           'options':{}}):
        
        """
        Initialize the CrystalGymEnv class.
        Args:
            data_path (str): The path to the data file.
            kwargs (dict): A dictionary containing the project, group, and experiment
        """
        super(CrystalGymEnv, self).__init__()

        ## Load the data
        self.data = pd.read_csv(data_path)
        # Define the action and observation space
        self.action_space = self.single_action_space = gym.spaces.Discrete(len(ELEMENTS_SMALL))
        self.observation_space =  self.single_observation_space = gym.spaces.Box(low=0, high=100, shape=(1,)) # Dummy observation space; actual space is a graph


        ## DFT Inputs
        pseudo_dir = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crysrl/crysrl/files/SSSP'
        pseudodict = pickle.load(open('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crysrl/crysrl/files/pseudodict.pkl', 'rb'))
        qe_inputs = {'prefix':"myprefix",'electron_maxstep':200,'outdir':'calculations','pseudo_dir': pseudo_dir, 'tstress':False,'tprnfor':False,'calculation':'scf', 
                    'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'fixed','smearing':'gaussian', 'mixing_mode':'plain', 
                    'mixing_beta':0.7,'degauss':0.001, 'nspin':1, 'ntyp': 1}

        profile = EspressoProfile(
                command = "mpirun --bind-to none -np 1 /home/mila/p/prashant.govindarajan/scratch/qe-7.1/bin/pw.x",
                pseudo_dir = pseudo_dir,
            )

        self.calculator = Espresso(profile = profile,
                                    pseudopotentials=pseudodict,
                                    input_data=qe_inputs, 
                                    kpts=(3,3,3), 
                                    directory= os.path.join('calculations', "_".join([kwargs['project'], kwargs['group'], kwargs['exp_name']])))
        ####
    
        self.state, _ = self.reset(kwargs['seed'], kwargs['options'])
        self.t = 0

    def reset(self, seed = 0, options = {}) :
        """
        Reset the environment.
        Returns:
            state (dict): The state of the environment.
        """
        info = {}
        self.index = 0
        self.ret = 0
        self.sample_ind = 0
        self.bfs_start = 0 
        self.t = 0
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        graph = build_crystal_graph(canonical_crystal, SPECIES_IND_INV)
        self.n_sites = graph.num_nodes()
        self.history = []
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
        graph.focus = self.traversal[self.t]
        graph.focus_list = self.traversal
        state = collate_function_crysrl(graph, p_hat = 1.12)

        lengths = torch.tensor(canonical_crystal.lattice.abc)
        angles = torch.tensor(canonical_crystal.lattice.angles)
        state.lengths_angles = torch.cat([lengths, angles])
        self.n_sites = state.num_nodes()
        self.history = []
        self.err_flag = 0

        if self.bfs_start >= self.n_sites:
            self.err_flag = 1
            return None
        self.traversal = torch.cat(list(bfs_nodes_generator(state, self.bfs_start)))
        try:
            assert len(self.traversal) == self.n_sites
        except:
            self.traversal = torch.tensor(list(range(self.n_sites)))
            self.err_flag = 1
        self.t = 0
        return state, info
    
    def compute_reward(self):
        """
        Compute the reward.
        Returns:
            reward (float): The reward.
        """

        canonical_crystal = self.render()
        # species = set([sp.name for sp in canonical_crystal.species])
        atoms = AseAtomsAdaptor.get_atoms(canonical_crystal)
        atoms.calc = self.calculator
        try:
            energy = atoms.get_potential_energy()
        except:
            pass
        try:        
            with open("/".join(['/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/cleanrl/calculations/'+"_".join([self.project, self.group, self.exp_name]), 'espresso.pwo']), 'r') as f:
                lines = f.read()
                tmp = lines.split('highest occupied, lowest unoccupied level (ev):')[-1].split()[:2]
                bg = float(tmp[1]) - float(tmp[0])
                if bg < 0.0:
                    bg = 0.0
                self.reward_model.train()
                success = True
                self.num_simulation_steps += 1
                dft_bg = bg
                reward = self.distance(self.p_hat, torch.tensor([bg]))
        except:
            reward = -1.0

        return reward

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
            reward = self.compute_reward()
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
        state_dict = {'frac_coords':np.array(frac_coords), 'atom_types':np.array(atomic_number.cpu()), 'lengths':np.array(lengths), 'angles':np.array(angles), 'num_atoms':num_atoms}
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
    