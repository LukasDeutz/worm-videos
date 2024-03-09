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

    log_dir, _, sweep_dir  = get_storage_dir('actuation_relaxation')
    
    h5, PG = load_data(sweep_dir, log_dir, h5_filename) 

    c0_arr = PG.c_from_k('c')
    lam0_arr = PG.v_from_k('lam')
    
    t = h5['t'][:] 
    shape = h5['FS']['r'].shape

    video_dir = video_dir / 'actuation_relaxation' / h5_filename.stem
    video_dir.mkdir(parent=True, exist_ok = True)
    
    for i, c0 in enumerate(c0_arr):
        for j, lam0 in enumerate(lam0_arr):
            
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
                video_dir / f'c0={c0}_lam0={lam0}', 
                add_trajectory = False, 
                draw_e3 = False, 
                n_arrows = 0.025)

def make_video_undulation_sweep():
    
    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=2.0_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=10.0.h5')    

    log_dir, _, sweep_dir  = get_storage_dir('undulation')
    
    h5, PG = load_data(sweep_dir, log_dir, h5_filename) 

    c0_arr = PG.c_from_k('c')
    lam0_arr = PG.v_from_k('lam')
    
    t = h5['t'][:] 
    shape = h5['FS']['r'].shape

    video_dir = video_dir / 'undulation' / h5_filename.stem
    video_dir.mkdir(parent=True, exist_ok = True)
    
    for i, c0 in enumerate(c0_arr):
        for j, lam0 in enumerate(lam0_arr):
            
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
                video_dir / f'c0={c0}_lam0={lam0}', 
                add_trajectory = False, 
                draw_e3 = False, 
                n_arrows = 0.025)

def generate_video(
        filepath: Path,
        r: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
        k: np.ndarray,
        t: np.ndarray):
    
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
            filepath, 
            add_trajectory = False, 
            draw_e3 = False, 
            n_arrows = 0.025)

        return
    
def showcase_dynamical_regimes():
    
    # Change file name
    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=2.0_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=10.0.h5')    

    h5, PG = load_data(h5_filename) 

    a_arr, b_arr = PG.v_from_key('a'), PG.v_from_key('b')
    
    t = h5['t'][:]
    #T = PG.base_parameter['T']    
    fps = int(1.0 / PG.base_parameter['dt'])

    # Regimes
    a1, b1 = 0.4, 0.005
    a2, b2 = 45, 0.1
    a3, b3 = 2200, 0.4
                        
    video_dir = video_dir / 'undulation' / h5_filename.stem
    video_dir.mkdir(parent=True, exist_ok = True)
    
    print(video_dir.resolve())

    for n, (ai, bi) in enumerate(zip([a1, a2, a3], [b1, b2, b3])):
        
        i = np.abs(a_arr - ai).min()
        j = np.abs(b_arr - bi).min()
                                
        l = np.ravel_multi_index((i,j), (len(a_arr), len(b_arr)))
    
        r = h5['FS']['r'][l, :]
        d1 = h5['FS']['d1'][l, :]
        d2 = h5['FS']['d2'][l, :]
        d3 = h5['FS']['d3'][l, :]
        k = h5['FS']['k'][l, :]
        
        filepath = video_dir / f'regime_{n+1}_a={ai}_b={bi}'
        
        generate_video(filepath, r, d1, d2, d3, k, t)
                
    return

if __name__ == '__main__':
    
    make_video_actuation_relaxation_sweep()
    
    
    


