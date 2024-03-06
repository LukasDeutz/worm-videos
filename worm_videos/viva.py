'''
Created on 6 Mar 2024

@author: amoghasiddhi
'''
from worm_studio import WormStudio
import pickle

from data_dirs import *
from video_dirs import video_dir

def make_video_actuation_relaxation():

    assert sim_dir_actu_rexa.is_dir()
        
    filepath = sim_dir_actu_rexa / 'raw_data_c=1.0_lam=1.0_tau_off_0.05_N=750_dt=0.01_T=5.0.h5'
        
    FS = pickle.load(open(filepath, 'rb'))
      
    studio = WormStudio(FS)

    video_path = video_dir / filepath.stem   

    studio.generate_clip(
        video_path, 
        add_trajectory = False, 
        draw_e3 = False, 
        n_arrows = 0.2)
      
    return      
          
if __name__ == '__main__':
    
    make_video_actuation_relaxation()
    
    
    

