"""CrystalGym Environment for Reinforcement Learning-based Crystal Design.

This module provides a Gymnasium-compatible environment for training RL agents
on crystal structure optimization tasks using DFT calculations.
"""

import os
import pickle
import subprocess
import time
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from ase.calculators.espresso import Espresso, EspressoProfile
from dgl.traversal import bfs_nodes_generator
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Kpoints

from crystal_gym.utils import cart_to_frac_coords
from crystal_gym.utils.create_graph import collate_function_crysrl
from crystal_gym.utils.data_utils import build_crystal, build_crystal_graph
from crystal_gym.utils.variables import (
    ELEMENTS_SMALL,
    ELEMENTS_MEDIUM,
    ELEMENTS_LARGE,
    SPECIES_IND_SMALL,
    SPECIES_IND_MEDIUM,
    SPECIES_IND_LARGE,
    SPACE_GROUP_TYPE,
    CUBIC_MINI,
)

# Physical constants
RYDBERG_TO_EV = 13.605691932782346  # Conversion factor from Rydberg to eV

# Environment constants
DEFAULT_VOCAB_SIZES = {
    'small': len(ELEMENTS_SMALL),
    'medium': len(ELEMENTS_MEDIUM),
    'large': len(ELEMENTS_LARGE)
}

# Error codes
ERROR_CONVERGENCE = 1
ERROR_CHARGE = 2
ERROR_CALCULATION = 3

# DFT calculation parameters for bulk modulus
STRAIN_POINTS = 5
VOLUME_SCALING_FACTORS = np.linspace(0.98, 1.02, 5)
SHEAR_STRAIN_RANGE = (-0.02, 0.02)

# Property-specific reward penalties for failed calculations
REWARD_PENALTIES = {
    'bm': -5.0,
    'density': -1.0,
    'bg': -1.0,
}

class CrystalGymEnv(gym.Env):
    def __init__(self, 
                 kwargs) -> None:
        
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
        self.vocab = self.env_options['vocab']
        if self.vocab == 'small':  # small action space: 18 elements
            self.action_space = self.single_action_space = gym.spaces.Discrete(len(ELEMENTS_SMALL))
            self.vocab_size = len(ELEMENTS_SMALL)
        elif self.vocab == 'medium': # medium action space: 30 elements
            self.action_space = self.single_action_space = gym.spaces.Discrete(len(ELEMENTS_MEDIUM))
            self.vocab_size = len(ELEMENTS_MEDIUM)
        elif self.vocab == 'large': # large action space: 50 elements
            self.action_space = self.single_action_space = gym.spaces.Discrete(len(ELEMENTS_LARGE))
            self.vocab_size = len(ELEMENTS_LARGE)

        # Dummy observation space; actual space is a graph
        self.observation_space =  self.single_observation_space = gym.spaces.Box(low=0, high=100, shape=(1,)) 

        ## DFT Inputs
        self.qe_inputs = kwargs['qe']
        self.pseudodict = pickle.load(open(kwargs['qe']['pseudodict'], 'rb'))
        pseudo_dir = kwargs['qe']['pseudo_dir']

        self.profile = EspressoProfile(
                command = f"mpirun --bind-to none -np 1 {kwargs['qe']['qe_dir']}/bin/pw.x",
                pseudo_dir = pseudo_dir,
            )
        
        self.agent = self.env_options['agent']
        self.state, _ = self.reset(self.env_options['seed'])
        self.t = 0


    def reset(self, 
            seed: Optional[int] = None, 
            options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.
        Returns:
            state (dict): The state of the environment.
        """
        random.seed(seed)
        info = {}
        if options is None:
            options = {}
        if self.env_options['mode'] == 'single':
            self.sample_ind = self.env_options['index']
        elif self.env_options['mode'] == 'cubic-mini':
            self.sample_ind = np.random.choice(CUBIC_MINI)

        cif_string = self.data.loc[self.sample_ind]['cif']
        self.space_grp = self.data.loc[self.sample_ind]['spacegroup.number']
        canonical_crystal = build_crystal(cif_string)
        if self.vocab == 'small':
            graph = build_crystal_graph(canonical_crystal, SPECIES_IND_SMALL, vocab_size = self.vocab_size, substitution = self.env_options['substitution'])
        elif self.vocab == 'medium':
            graph = build_crystal_graph(canonical_crystal, SPECIES_IND_MEDIUM, vocab_size = self.vocab_size, substitution = self.env_options['substitution'])
        elif self.vocab == 'large':
            graph = build_crystal_graph(canonical_crystal, SPECIES_IND_LARGE, vocab_size = self.vocab_size, substitution = self.env_options['substitution'])

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
                if self.vocab == 'small':
                    state.replace(i, Element.from_Z(SPECIES_IND_SMALL[graph.ndata['atomic_number'][i].item()]))
                elif self.vocab == 'medium':
                    state.replace(i, Element.from_Z(SPECIES_IND_MEDIUM[graph.ndata['atomic_number'][i].item()]))
                elif self.vocab == 'large':
                    state.replace(i, Element.from_Z(SPECIES_IND_LARGE[graph.ndata['atomic_number'][i].item()]))
        self.state = state
        return state, info

  
    def calculate_bm(self, atoms) -> Tuple[Optional[float], int]:
        """Calculate the bulk modulus using equation of state fitting.
        
        Args:
            atoms: ASE Atoms object representing the crystal structure
            
        Returns:
            Tuple of (bulk_modulus, error_flag):
                - bulk_modulus: Calculated bulk modulus in GPa, or None if failed
                - error_flag: 0 for success, 1 for calculation error, 2 for parsing error
        """
        lengths = []
        energies = []
        
        # Calculate energy-volume curve
        for factor in VOLUME_SCALING_FACTORS:
            scaled_atoms = atoms.copy()
            scaled_atoms.set_cell(atoms.get_cell() * factor**(1/3), scale_atoms=True)
            scaled_atoms.calc = atoms.calc
            
            try:
                energy = scaled_atoms.get_potential_energy() / RYDBERG_TO_EV
                volume = scaled_atoms.get_volume()
                lengths.append(volume ** (1/3))
                energies.append(energy)
            except Exception:
                return None, ERROR_CALCULATION
        
        # Write data file for EOS fitting
        calc_dir = os.path.join('calculations', self.run_name)
        os.makedirs(calc_dir, exist_ok=True)
        
        with open(os.path.join(calc_dir, 'length_energy.dat'), 'w') as f:
            for v, e in zip(lengths, energies):
                f.write(f"{v:.6f} {e:.6f}\n")
        
        # Write input file for ev.x
        spg_type = SPACE_GROUP_TYPE[self.space_grp]
        with open(os.path.join(calc_dir, 'ev.in'), 'w') as f:
            f.write("Ang\n")
            f.write(f"{spg_type}\n")  # Use 'noncubic' to treat input as volumes
            f.write("4\n")  # Murnaghan EOS
            f.write(os.path.join(calc_dir, 'length_energy.dat') + "\n")
            f.write(os.path.join(calc_dir, 'ev.txt') + "\n")

        # Run ev.x to fit EOS
        path = os.path.join(calc_dir, 'ev.in')
        result = subprocess.run(
            f"mpirun --bind-to none -np 1 {self.qe_inputs['qe_dir']}/bin/ev.x < {path}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None, ERROR_CALCULATION

        # Parse bulk modulus from output
        try:
            with open(os.path.join(calc_dir, 'ev.txt'), 'r') as f:
                lines = f.readlines()
                if len(lines) < 3:
                    return None, ERROR_CALCULATION
                    
                string = lines[2].split()[7]
                if 'GPa' in string:
                    bm = float(lines[2].split()[6].split('=')[1])
                else:
                    bm = float(lines[2].split()[7])
                return bm, 0
        except (ValueError, IndexError):
            return None, ERROR_CHARGE
    
    def calculate_band_gap(self, atoms) -> Tuple[Optional[float], int]:
        """Calculate the band gap from DFT calculation output.
        
        Args:
            atoms: ASE Atoms object (used to trigger calculation)
            
        Returns:
            Tuple of (band_gap, error_flag):
                - band_gap: Calculated band gap in eV, or None if failed
                - error_flag: 0 for success, 1 for convergence error, 2 for charge error, 3 for calculation error
        """
        try:
            atoms.get_potential_energy()
        except Exception:
            pass  # Energy calculation may fail, but we can still read band gap
            
        calc_dir = os.path.join('calculations', self.run_name)
        output_file = os.path.join(calc_dir, 'espresso.pwo')
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                
            if 'convergence NOT achieved after' in content:
                return None, ERROR_CONVERGENCE
            elif 'charge is wrong' in content:
                return None, ERROR_CHARGE
            
            # Extract band gap from output
            tmp = content.split('highest occupied, lowest unoccupied level (ev):')[-1].split()[:2]
            bg = float(tmp[1]) - float(tmp[0])
            bg = max(0.0, bg)  # Ensure non-negative band gap
            return bg, 0
            
        except (FileNotFoundError, ValueError, IndexError):
            return None, ERROR_CALCULATION
    
    def calculate_density(self, atoms) -> Tuple[Optional[float], int]:
        """Calculate the density from DFT calculation output.
        
        Args:
            atoms: ASE Atoms object (used to trigger calculation)
            
        Returns:
            Tuple of (density, error_flag):
                - density: Calculated density in g/cm³, or None if failed
                - error_flag: 0 for success, 1 for convergence error, 2 for charge error
        """
        try:
            atoms.get_potential_energy()
        except Exception:
            pass  # Energy calculation may fail, but we can still read density
            
        calc_dir = os.path.join('calculations', self.run_name)
        output_file = os.path.join(calc_dir, 'espresso.pwo')
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            density = float(content.split('density =')[1].split()[0])
            return density, 0
            
        except (FileNotFoundError, ValueError, IndexError):
            return None, ERROR_CALCULATION

    
    def compute_reward(self) -> Tuple[float, Optional[float], int, Optional[float]]:
        """Compute the reward based on the target property.
        
        Returns:
            Tuple of (reward, property_value, error_flag, simulation_time):
                - reward: Computed reward value
                - property_value: Calculated property value (band gap, bulk modulus, etc.)
                - error_flag: 0 for success, >0 for various error types
                - simulation_time: Time taken for DFT calculation in seconds
        """
        error_flag = 0
        if self.agent == "MEGNetRL":
            canonical_crystal = self.render()
        elif self.agent == "CHGNetRL":
            canonical_crystal = self.state
        else:
            raise ValueError(f"Unknown agent type: {self.agent}")
            
        # Set up DFT calculation
        atoms = AseAtomsAdaptor.get_atoms(canonical_crystal)
        nbnd = int(np.ceil(sum(atoms.get_atomic_numbers()) // 2 * 1.2))
        self.qe_inputs.update({'nbnd': nbnd})
        kpts = Kpoints.automatic_density(canonical_crystal, kppa=self.qe_inputs['kppa']).kpts[0]
        
        calc_dir = os.path.join('calculations', self.run_name)
        os.makedirs(calc_dir, exist_ok=True)
        
        # Initialize the calculator
        atoms.calc = Espresso(
            profile=self.profile,
            pseudopotentials=self.pseudodict,
            input_data=self.qe_inputs,
            kpts=kpts,
            directory=calc_dir
        )
        
        # Calculate target property
        property_type = self.env_options['property']
        start_time = time.time()
        
        if property_type == 'bm':
            cell_dm = canonical_crystal.lattice.a
            bm, error_flag = self.calculate_bm(atoms, cell_dm)
            end_time = time.time()
            
            if error_flag == 0:
                reward = -np.abs(self.env_options['p_hat'] - bm) / self.env_options['p_hat']
                sim_time = end_time - start_time
                if self.env_options.get('reward_min') and reward < self.env_options['reward_min']:
                    reward = self.env_options['reward_min']
                return reward, bm, error_flag, sim_time
            else:
                return REWARD_PENALTIES['bm'], None, error_flag, None
                
        
        elif self.env_options['property'] == 'density':
            start_time = time.time()
            density, error_flag = self.calculate_density(atoms)
            end_time = time.time()

            if error_flag == 0:
                reward = self.distance(self.env_options['p_hat'], torch.tensor([density]), self.env_options['p_hat']).item()
                sim_time = end_time - start_time
                return reward, density, error_flag, sim_time
            else:
                return REWARD_PENALTIES['density'], None, error_flag, None
        
        else:  # Default to band gap calculation
            bg, error_flag = self.calculate_band_gap(atoms)
            end_time = time.time()
            
            if error_flag == 0:
                reward = self.distance(self.env_options['p_hat'], torch.tensor([bg])).item()
                sim_time = end_time - start_time
                return reward, bg, error_flag, sim_time
            else:
                return REWARD_PENALTIES['bg'], None, error_flag, None
            
    
    def distance(self, target: float, predicted: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Compute the exponential distance between two vectors.
        Args:
            target: Target value
            predicted: Predicted value(s) as tensor
            beta: Scaling parameter for the exponential
            
        Returns:
            torch.Tensor: Exponential distance value
        """
        try:
            return torch.exp(-(target - predicted)**2 / beta)[0]
        except IndexError:
            return torch.exp(-(target - predicted)**2 / beta)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        Args:
            action: Index of the atomic species to substitute (0 to vocab_size-1)
            
        Returns:
            Tuple containing:
                - state: Updated crystal graph or structure
                - reward: Reward value (0 during episode, computed at end)
                - terminated: Whether the episode is terminated
                - truncated: Whether the episode is truncated
                - info: Additional information dictionary
        """
        info = {}
        index_curr_focus = self.traversal[self.t]
        
        # Completion/Substitution based on agent type
        if self.agent == "MEGNetRL":
            atomic_number = deepcopy(self.state.ndata['atomic_number'])
            atomic_number[index_curr_focus] = torch.tensor(action)
            next_observations = deepcopy(self.state)
            next_observations.ndata['atomic_number'] = atomic_number
            if self.t+1 < self.n_sites:
                next_observations.focus = torch.tensor([self.traversal[self.t+1]], device='cuda')
            else:
                next_observations.focus = torch.tensor([20], device='cuda')  # dummy focus (assuming there are no more than 20 atoms)
            self.state = next_observations
            
        elif self.agent == "CHGNetRL":
            next_observations = deepcopy(self.state)
            species_mapping = {
                'small': SPECIES_IND_SMALL,
                'medium': SPECIES_IND_MEDIUM,
                'large': SPECIES_IND_LARGE
            }
            new_element = Element.from_Z(species_mapping[self.vocab][action.item()])
            next_observations.replace(index_curr_focus, new_element)
            self.state = next_observations
        else:
            raise ValueError(f"Unknown agent type: {self.agent}")
            
        self.t += 1

        # Check if episode is complete
        if self.t == self.n_sites:
            terminated = truncated = True
            reward, property_value, error_flag, sim_time = self.compute_reward()
            
            info['final_info'] = [{
                'episode': {
                    'r': reward,
                    'error_flag': error_flag
                }
            }]
            
            if property_value is not None:
                info['final_info'][0]['episode'][self.env_options['property']] = property_value
                if sim_time is not None:
                    info['final_info'][0]['episode']['sim_time'] = sim_time
        else:
            terminated = truncated = False
            reward = 0.0

        return self.state, reward, terminated, truncated, info
    
    def graph_to_dict_complete(self, observations) -> Dict[str, np.ndarray]:
        """Convert the graph to a complete dictionary representation.
        
        Args:
            observations: Graph observations containing atomic data
            
        Returns:
            Dictionary containing:
                - frac_coords: Fractional coordinates of atoms
                - atom_types: Atomic numbers
                - lengths: Lattice lengths
                - angles: Lattice angles
                - num_atoms: Number of atoms
        """
        atomic_number = deepcopy(observations.ndata['atomic_number'])
        position = deepcopy(observations.ndata['position'])
        lengths = deepcopy(observations.lengths_angles_focus.cpu()[0][:3])
        angles = deepcopy(observations.lengths_angles_focus.cpu()[0][3:6])
        num_atoms = atomic_number.shape[0]
        
        frac_coords = cart_to_frac_coords(
            position.to(dtype=torch.float32).cpu(),
            lengths.unsqueeze(0),
            angles.unsqueeze(0),
            num_atoms
        )
        
        return {
            'frac_coords': np.array(frac_coords),
            'atom_types': np.array(atomic_number.cpu()),
            'lengths': np.array(lengths),
            'angles': np.array(angles),
            'num_atoms': num_atoms
        }
    
    def graph_to_dict(self, observation) -> Dict[str, torch.Tensor]:
        """Convert the graph to a dictionary for the RL replay buffer.
        
        Args:
            observation: Graph observation containing atomic and edge data
            
        Returns:
            Dictionary containing graph components for replay buffer storage
        """
        state = {}
        focus = observation.focus.to(device='cuda')
            
        state['atomic_number'] = torch.cat([
            observation.ndata['atomic_number'].to(device='cuda'),
            focus
        ])
        state['coordinates'] = observation.ndata['position']
        state['edges'] = observation.edges()
        state['efeat'] = observation.edata['e_feat']
        state['etype'] = observation.edata['etype'].squeeze()
        state['laf'] = observation.lengths_angles_focus.squeeze()
        
        return state
    
    def to_struct(self, state_dict: Dict[str, np.ndarray]) -> Structure:
        """Convert the dictionary to a pymatgen Structure.
        
        Args:
            state_dict: Dictionary containing crystal structure data
            
        Returns:
            pymatgen Structure object
        """
        lengths = state_dict['lengths'].tolist()
        angles = state_dict['angles'].tolist()
        lattice_params = lengths + angles
        atomic_number = state_dict['atom_types']
        
        species_mapping = {
            'small': SPECIES_IND_SMALL,
            'medium': SPECIES_IND_MEDIUM,
            'large': SPECIES_IND_LARGE
        }
        
        atom_types = [
            species_mapping[self.vocab][int(atomic_number[j])]
            for j in range(atomic_number.shape[0])
        ]
        
        coords = state_dict['frac_coords']
        return Structure(
            lattice=Lattice.from_parameters(*lattice_params),
            species=atom_types,
            coords=coords
        )

    def render(self, mode: str = "crystal") -> Structure:
        """Render the current crystal structure.
        
        Args:
            mode: Rendering mode (currently only "human" supported)
            
        Returns:
            pymatgen Structure object representing the current crystal
        """
        crystal_dict = self.graph_to_dict_complete(self.state)
        return self.to_struct(crystal_dict)
    
    def get_obs(self) -> Any:
        """Get the current observation.
        
        Returns:
            Current state (graph or structure depending on agent type)
        """
        return self.state
    
    def close(self) -> None:
        """Close the environment and clean up resources.
        
        Currently resets the environment to initial state.
        """
        self.reset()
    
