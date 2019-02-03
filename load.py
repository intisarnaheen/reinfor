import numpy as np
array_loaded = np.load('T_new.npy')
print(array_loaded)
t_s= array_loaded.shape
t_dim = array_loaded.ndim
print(t_s)
print(t_dim)
