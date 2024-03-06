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

log_dir_actu_rexa = data_path_undu / 'logs'
sim_dir_actu_rexa = data_path_undu / 'simulations'
sweep_dir_actu_rexa = data_path_undu / 'parameter_sweeps'

for Dir in [log_dir_undu, sim_dir_undu, sweep_dir_undu, 
    log_dir_actu_rexa, sim_dir_actu_rexa, sweep_dir_actu_rexa]:

    assert Dir.is_dir()


