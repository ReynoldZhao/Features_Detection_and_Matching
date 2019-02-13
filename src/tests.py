import numpy as np
from scipy import ndimage
from scipy.ndimage import filters

a = np.random.randint(0,15,(10,10))
newmax = ndimage.maximum_filter(a,size=(2,2),mode='reflect')
b = a == newmax
print(a)
print(newmax)
print(b)