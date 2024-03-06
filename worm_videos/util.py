'''
Created on 6 Mar 2024

@author: amoghasiddhi
'''
from types import SimpleNamespace

def convert_to_FSN(FS):
    
    return SimpleNamespace(**{
        'x': FS.x,
        'e1': FS.e1,
        'e2': FS.e2,
        'e3': FS.e3,
        'times': FS.times})
    
    
