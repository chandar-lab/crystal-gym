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
from crystal_gym.utils.create_graph import collate_function_crysrl
from crystal_gym.utils import cart_to_frac_coords
from ase.calculators.espresso import Espresso, EspressoProfile
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core import Structure, Lattice
from crystal_gym.utils.data_utils import build_crystal, build_crystal_graph
from crystal_gym.utils.variables import ELEMENTS_SMALL, SPECIES_IND_INV, SPECIES_IND_SMALL, SPACE_GROUP_TYPE, SPECIES_IND_SMALL_INV
from dgl.traversal import bfs_nodes_generator
from crystal_gym.utils.variables import CUBIC_INDS_VAL, CUBIC_VAL_FIVE, CUBIC_MINI
from pymatgen.core import Element
import subprocess

RY_CONST = 13.605691932782346

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
                command = f"mpirun --bind-to none -np 1 {kwargs['qe']['qe_dir']}/bin/pw.x",
                pseudo_dir = pseudo_dir,
            )
        
        self.agent = self.env_options['agent']
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
        elif self.env_options['mode'] == 'cubic-all':
            self.sample_ind = np.random.choice(CUBIC_INDS_VAL)
        elif self.env_options['mode'] == 'cubic-five':  
            self.sample_ind = np.random.choice(CUBIC_VAL_FIVE)
        elif self.env_options['mode'] == 'cubic-mini':
            self.sample_ind = np.random.choice(CUBIC_MINI)

        cif_string = self.data.loc[self.sample_ind]['cif']
        self.space_grp = self.data.loc[self.sample_ind]['spacegroup.number']
        canonical_crystal = build_crystal(cif_string)

        graph = build_crystal_graph(canonical_crystal, SPECIES_IND_SMALL, substitution = self.env_options['substitution'])

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

        if self.agent=='CHGNetRL':
            assert self.env_options['substitution'] == True, "CHGNetRL only works with substitution"
            state = canonical_crystal
            for i in range(self.n_sites):
                state.replace(i, Element.from_Z(SPECIES_IND_SMALL[graph.ndata['atomic_number'][i].item()]))
        self.state = state
        return state, info

    def calculate_sm(self, atoms):
        """
        Calculate the shear modulus.
        """

        strains = np.linspace(-0.02, 0.02, 5) # 5 points

        energies = []
        stresses = []

        for strain in strains:
            deformed_atoms = atoms.copy()
            cell = deformed_atoms.get_cell()
            
            # Apply shear strain (e.g., xy component)
            deformation = np.eye(3)
            deformation[0, 1] = strain
            cell = np.dot(cell, deformation)
            
            deformed_atoms.set_cell(cell, scale_atoms=True)
            deformed_atoms.calc = atoms.calc
            
            energy = deformed_atoms.get_potential_energy()
            stress = deformed_atoms.get_stress()
            
            energies.append(energy)
            stresses.append(stress)
        #breakpoint()
        # Calculate shear modulus (C44 for cubic crystals)
        # volume = atoms.get_volume()
        shear_stresses = [stress[3] for stress in stresses]  # xy component
        
        slope, _ = np.polyfit(strains, shear_stresses, 1)
        shear_modulus = slope 
        
        return shear_modulus

    def calculate_bm(self, atoms, celldm):
        """
        Calculate the bulk modulus.
        Args:
            atoms (ase.Atoms): The atoms object.
        Returns:
            bm (float): The bulk modulus.
        """
        lengths = []
        energies = []
        for factor in np.linspace(0.98, 1.02, 5):
            scaled_atoms = atoms.copy()
            scaled_atoms.set_cell(atoms.get_cell() * factor**(1/3), scale_atoms=True)
            scaled_atoms.calc = atoms.calc
            try:
                energy = scaled_atoms.get_potential_energy() / RY_CONST
            except:
                return None, 1
            volume = scaled_atoms.get_volume()
            
            lengths.append(volume ** (1/3))
            energies.append(energy)
        
        with open(os.path.join('calculations', self.run_name, 'length_energy.dat'), 'w') as f:
            for v, e in zip(lengths, energies):
                f.write(f"{v:.6f} {e:.6f}\n")
        
        spg_type = SPACE_GROUP_TYPE[self.space_grp]
        with open(os.path.join('calculations', self.run_name, 'ev.in'), 'w') as f:
            f.write("Ang\n")
            f.write(f"{spg_type}\n")  # Use 'noncubic' to treat input as volumes
            f.write("4\n")  # Murnaghan EOS
            f.write(os.path.join('calculations', self.run_name, 'length_energy.dat') + "\n")
            f.write(os.path.join('calculations', self.run_name, 'ev.txt') + "\n")

        path = os.path.join('calculations', self.run_name, 'ev.in')
        subprocess.run(f"mpirun --bind-to none -np 1 {self.qe_inputs['qe_dir']}/bin/ev.x < {path}", shell=True)

        with open(os.path.join('calculations', self.run_name, 'ev.txt'), 'r') as f:
            lines = f.readlines()
            string = lines[2].split()[7]

            try:
                if 'GPa' in string:
                    bm = float(lines[2].split()[6].split('=')[1])
                else:
                    bm = float(lines[2].split()[7])
            except ValueError:
                return None, 2

        return bm, 0

    
    def compute_reward(self):
        """
        Compute the reward.
        Returns:
            reward (float): The reward.
        """
        error_flag = 0
        if self.agent == "MEGNetRL":
            canonical_crystal = self.render()
        elif self.agent == "CHGNetRL":
            canonical_crystal = self.state
        atoms = AseAtomsAdaptor.get_atoms(canonical_crystal)
        nbnd = int(np.ceil(sum(atoms.get_atomic_numbers()) // 2 * 1.2))
        self.qe_inputs.update({'nbnd':nbnd})
        kpts = Kpoints.automatic_density(canonical_crystal, kppa = self.qe_inputs['kppa']).kpts[0]
        # breakpoint()
        
        atoms.calc = Espresso(profile = self.profile,
                                pseudopotentials=self.pseudodict,
                                input_data=self.qe_inputs, 
                                kpts=kpts, 
                                directory= os.path.join('calculations', self.run_name))
        if self.env_options['property'] == 'bm':
            cell_dm = canonical_crystal.lattice.a
            start_time = time.time()
            bm, error_flag = self.calculate_bm(atoms, cell_dm)
            end_time = time.time()
            if error_flag == 0:
                reward = - np.abs(self.env_options['p_hat'] - bm) / self.env_options['p_hat']
                sim_time = end_time - start_time
                if self.env_options['reward_min'] and reward < self.env_options['reward_min']:
                    reward = self.env_options['reward_min']
                return reward, bm, error_flag, sim_time
            else:
                return -5.0, None, error_flag, None
                
        elif self.env_options['property'] == 'sm':
            start_time = time.time()
            sm = self.calculate_sm(atoms)
            end_time = time.time()
            reward = - np.abs(self.env_options['p_hat'] - sm) / self.env_options['p_hat']
            sim_time = end_time - start_time
            return reward, sm, error_flag, sim_time
        else:
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
        index_curr_focus = self.traversal[self.t]
        if self.agent == "MEGNetRL":
            atomic_number = deepcopy(self.state.ndata['atomic_number'])
            atomic_number[index_curr_focus] = torch.tensor(action)
            next_observations = deepcopy(self.state)
            next_observations.ndata['atomic_number'] = atomic_number
            self.state = deepcopy(next_observations)
            self.t += 1
            
        elif self.agent == "CHGNetRL":
            next_observations = deepcopy(self.state)
            new_element = Element.from_Z(SPECIES_IND_SMALL[action.item()])
            next_observations.replace(index_curr_focus, new_element)
            self.state = deepcopy(next_observations)
            self.t += 1

        if self.t == self.n_sites:
            terminated = truncated = True
            # self.t = 0
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
    
    def graph_to_dict(self, observation):
        """
        Convert the graph to a dictionary for the RL replay buffer
        """
        state = {}
        focus = observation.focus.to(device = 'cuda')
        if focus.item() == -10000:
            focus = torch.tensor([20])
        state['atomic_number'] = torch.cat([observation.ndata['atomic_number'].to(device = 'cuda'), focus])
        state['coordinates'] = observation.ndata['position']
        state['edges'] = observation.edges()
        state['efeat'] = observation.edata['e_feat']
        state['etype'] = observation.edata['etype'].squeeze()
        state['laf'] = observation.lengths_angles_focus.squeeze()
        # state['index'] = observation.inds
        return state
    
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
    
