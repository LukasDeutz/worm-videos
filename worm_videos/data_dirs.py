'''
Created on 6 Mar 2024

@author: amoghasiddhi
'''

from pathlib import Path


# Paths to data directory in the minimal-worm repository

data_path = Path('../../minimal-worm/minimal_worm/experiments/results/data')

data_path_undu = data_path / 'undulation'
data_path_actu_rexa = data_path / 'actuation_relaxation'

log_dir_undu = data_path_undu / 'logs'
sim_dir_undu = data_path_undu / 'simulations'
sweep_dir_undu = data_path_undu / 'parameter_sweeps'

log_dir_actu_rexa = data_path_actu_rexa/ 'logs'
sim_dir_actu_rexa = data_path_actu_rexa / 'simulations'
sweep_dir_actu_rexa = data_path_actu_rexa / 'parameter_sweeps'

for Dir in [log_dir_undu, sim_dir_undu, sweep_dir_undu, 
    log_dir_actu_rexa, sim_dir_actu_rexa, sweep_dir_actu_rexa]:

    assert Dir.is_dir()


storage_dir = Path('/home/lukas/storage/minimal-worm/results/')  
base_path = Path('/home/lukas/git/minimal-worm/minimal_worm/experiments/')

storage_dir = storage_dir.relative_to('/home/lukas/')
base_path = base_path.relative_to('/home/lukas/')
relative_path = Path(*(['../']*len(base_path.parts)))

storage_dir = relative_path / storage_dir 
storage_dir.mkdir(parents = True, exist_ok = True)

def get_storage_dir(experiment_type):

    storage_dir_exp = storage_dir / experiment_type
    
    assert storage_dir_exp.isdir()
                                                                                 
    log_dir = storage_dir / 'logs'
    sim_dir = storage_dir / 'simulations'
    sweep_dir= storage_dir / 'parameter_sweeps'
    
    return log_dir, sim_dir, sweep_dir





