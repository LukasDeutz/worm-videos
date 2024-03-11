'''
Created on 6 Mar 2024

@author: amoghasiddhi
'''
from typing import Optional
from worm_studio import WormStudio
import pickle
from scipy.optimize import curve_fit

from parameter_scan import ParameterGrid
import h5py
import numpy as np

from data_dirs import *
from video_dirs import video_dir
from types import SimpleNamespace
from threading import enumerate

def fang_yen_data():
    '''
    Undulation wavelength, amplitude, frequencies for different viscosities
    from Fang-Yen 2010 paper    
    '''
    # Experimental Fang Yeng 2010
    mu_arr = 10**(np.array([0.000, 0.966, 2.085, 2.482, 2.902, 3.142, 3.955, 4.448])-3) # Pa*s            
    lam_arr = np.array([1.516, 1.388, 1.328, 1.239, 1.032, 0.943, 0.856, 0.799])        
    f_arr = [1.761, 1.597, 1.383, 1.119, 0.790, 0.650, 0.257, 0.169] # Hz
    A_arr = [2.872, 3.126, 3.290, 3.535, 4.772, 4.817, 6.226, 6.735]
    
    return mu_arr, lam_arr, f_arr, A_arr

#===============================================================================
# Fits
#===============================================================================

def fang_yen_fit(return_param = False):
    '''
    Fit sigmoids to fang yen data
    '''
    mu_arr, lam_arr, f_arr, A_arr = fang_yen_data()

    log_mu_arr = np.log10(mu_arr)

    # Define the sigmoid function
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    # Fit the sigmoid to wavelength
    popt_lam, _ = curve_fit(sigmoid, log_mu_arr,lam_arr)
    lam_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_lam)

    # Fit the sigmoid to frequency
    popt_f, _ = curve_fit(sigmoid, log_mu_arr, f_arr)
    f_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_f)

    # Fit the sigmoid to amplitude
    a0 = 3.95
    b0 = 0.12    
    c0 = 2.13
    d0 = 2.94
    p0 = [a0, b0, c0, d0] 
    popt_A, _ = curve_fit(sigmoid, log_mu_arr, A_arr, p0=p0)
    A_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_A)
    
    if not return_param:
        return lam_sig_fit, f_sig_fit, A_sig_fit
    return lam_sig_fit, f_sig_fit, A_sig_fit, popt_lam, popt_f, popt_A 



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
        t: np.ndarray,
        f: Optional[float] = None):
    
        FS = SimpleNamespace(**{
            'r': r, 
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'k': k,
            't': t
            }
        )

        studio = WormStudio(FS, f)

        studio.generate_clip(
            filepath, 
            add_trajectory = False, 
            draw_e3 = False, 
            n_arrows = 0.025,
            f)

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

def showcase_frequency_adpatation():
    
    h5_filename = Path('raw_data_a_arr=[6.0300e-02 9.5494e+02 1.1370e+02]_b_arr=[0.009 0.009 0.001]_A=2.94_lam=1.46_T=5.0_N=750_dt=0.001.h5')
    
    h5, PG = load_data(h5_filename) 

    a_arr, b_arr = PG.v_from_key('a'), PG.v_from_key('b')
    
    t = h5['t'][:]
    #T = PG.base_parameter['T']    

    # Access grid parameters
    a_arr = PG.v_from_key('a')
    b_arr = PG.v_from_key('b')
                
    #===============================================================================
    # Choose operating points
    #===============================================================================
    
    N = int(1e2)

    # Viscosity range                                    
    mu0, mu1 = 10**(-3.0), 10**(+1.2)
     
    f_sig_fit = fang_yen_fit()[1]
    
    # With frequency modulation from experimental data
    f0 = f_sig_fit(np.log10(mu0))
    f1 = f_sig_fit(np.log10(mu1))
                                        
    # Geometric parameter
    L0 = 1130 * 1e-6
    Rmax = 32 * 1e-6    
    # Material parameter
    B = 9.5*1e-14
    xi = 0.005    
    # Update to be consistent with model fit                           
    ct = 2.05
    mu0 = 1e-3
    # C elegans Operating point in water
    a0 = mu0 * ct * L0 **4 / B * f0
    b0 = xi * f0

    a1 = mu1 * ct * L0 **4 / B * f0
    b1 = xi * f0
    
    a2 = mu1 * ct * L0 **4 / B * f1
    b2 = xi * f1
                
    video_dir = video_dir / 'undulation' / 'frequency_adaptation'
    video_dir.mkdir(parent=True, exist_ok = True)
    
    print(video_dir.resolve())

    for n, (ai, bi, f) in enumerate(zip([a0, a1, a2], [b0, b1, b2], [f0, f0, f1])):
        
        i = np.abs(a_arr - ai).min()
        j = np.abs(b_arr - bi).min()
                                
        l = np.ravel_multi_index((i,j), (len(a_arr), len(b_arr)))
    
        r = h5['FS']['r'][l, :]
        d1 = h5['FS']['d1'][l, :]
        d2 = h5['FS']['d2'][l, :]
        d3 = h5['FS']['d3'][l, :]
        k = h5['FS']['k'][l, :]
        
        filepath = video_dir / f'{n}'
        
        generate_video(filepath, r, d1, d2, d3, k, t, f)

    return
    
if __name__ == '__main__':
    
    #make_video_actuation_relaxation_sweep()
    showcase_frequency_adpatation()
    
    
    


