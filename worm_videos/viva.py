'''
Created on 6 Mar 2024

@author: amoghasiddhi
'''
from worm_studio import WormStudio
import pickle


from parameter_scan import ParameterGrid
import h5py
import numpy as np

from data_dirs import *
from video_dirs import video_dir
from types import SimpleNamespace
from threading import enumerate

def load_data(sweep_dir, log_dir, filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))


    return h5, PG

def make_video_actuation_relaxation_():

    assert sim_dir_actu_rexa.is_dir()
        
    filepath = sim_dir_actu_rexa / 'raw_data_c=1.0_lam=1.0_tau_off_0.05_N=750_dt=0.01_T=5.0.h5'
        
    FS = pickle.load(open(filepath, 'rb'))
      
    studio = WormStudio(FS)

    video_path = video_dir / 'actuation_relaxation' / filepath.stem   

    studio.generate_clip(
        video_path, 
        add_trajectory = False, 
        draw_e3 = False, 
        n_arrows = 0.025)
      
    return      

def make_video_actuation_relaxation_sweep():

    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=1.5_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=5.0.h5')    

    log_dir, _, sweep_dir  = get_storage_dir('actuation_relaxation')[2]
    
    h5, PG = load_data(sweep_dir, log_dir, h5_filename) 

    c_arr = PG.c_from_k('lam')
    lam_arr = PG.v_from_k('lam')
    
    t = h5['t'][:] 
    shape = h5['FS']['r'].shape

    video_dir = video_dir / 'actuation_relaxation' / h5_filename.stem
    video_dir.mkdir(parent=True, exist_ok = True)
    
    for i, c in enumerate(c_arr):
        for j, lam in enumerate(lam_arr):
            
            k = np.ravel_multi_index((i,j), shape)
            
            r = h5['FS']['r'][k, :]
            d1 = h5['FS']['d1'][k, :]
            d2 = h5['FS']['d2'][k, :]
            d3 = h5['FS']['d3'][k, :]
            k = h5['FS']['k'][k, :]
        
            FS = SimpleNamespace(**{
                'r': r, 
                'd1': d1,
                'd2': d2,
                'd3': d3,
                'k': k,
                't': t
                }
            )

            studio = WormStudio(FS)

            studio.generate_clip(
                video_dir / f'c={c}_lam={lam}', 
                add_trajectory = False, 
                draw_e3 = False, 
                n_arrows = 0.025)
    
if __name__ == '__main__':
    
    make_video_actuation_relaxation_sweep()
    
    
    


