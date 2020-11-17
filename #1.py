#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Calculating the mean


# In[ ]:


fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
m = sum(fluxes)/len(fluxes)
print(m)


# In[ ]:


from statistics import mean
fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
m = mean(fluxes)
print(m)


# In[ ]:


def calculate_mean(data):
  mean = sum(data)/len(data)
  return mean

if __name__ == '__main__':
  m = calculate_mean([1,2.2,0.3,3.4,7.9])
  print(mean)


# In[ ]:


#Numpy Arrays


# In[ ]:


import numpy as np
fluxes = np.array([23.3, 42.1, 2.0, -3.2, 55.6])
m = np.mean(fluxes)
print(m)


# In[ ]:


import numpy as np
fluxes = np.array([23.3, 42.1, 2.0, -3.2, 55.6])
print(np.size(fluxes)) # length of array
print(np.std(fluxes))  # standard deviation


# In[ ]:


#Readin Strings from CSV Files


# In[ ]:


data = []
for line in open('data.csv'):
  data.append(line.strip().split(','))

print(data)

#nested for loop
data = []
for line in open('data.csv'):
  row = []
  for col in line.strip().split(','):
    row.append(float(col))
  data.append(row)

print(data)

#NumPy nested for loop
import numpy as np

data = []
for line in open('data.csv'):
  data.append(line.strip().split(','))

data = np.asarray(data, float)
print(data)

#best option
import numpy as np
data = np.loadtxt('data.csv', delimiter=',')
print(data)


# In[ ]:


#Mean of 1D array
import numpy as np

def calc_stats(filename):
  data = np.loadtxt(filename, delimiter=',')
 
  mean = np.mean(data)
  median = np.median(data)

  return np.round(mean, 1), np.round(median, 1)


# In[ ]:


#element wise operations
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise multiplication 
print(a*2)

# Element-wise summation 
print(a + b)

# Element-wise product 
print(a*b)


# In[ ]:


#NumPy array operations
import numpy as np

a = np.array([[1,2,3], [4,5,6]])  # 2x3 array

# Print first row of a:
print(a[0,:])

# Print second column of a:
print(a[:,1])


# In[ ]:


#Mean of a set of signals
import numpy as np

def mean_datasets(filenames):
  n = len(filenames)
  if n > 0:
    data = np.loadtxt(filenames[0], delimiter=',')
    for i in range(1,n):
      data += np.loadtxt(filenames[i], delimiter=',')
    
    # Mean across all files:
    data_mean = data/n
     
    return np.round(data_mean, 1)


# In[ ]:


#Flexible Image Transport System (FITS) files


# In[ ]:


from astropy.io import fits
hdulist = fits.open('image0.fits')
hdulist.info()


# In[ ]:


from astropy.io import fits

hdulist = fits.open('image0.fits')
data = hdulist[0].data

print(data.shape)


# In[2]:


from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('image0.fits')
data = hdulist[0].data

# Plot the 2D array
plt.imshow(data, cmap=plt.cm.viridis)
plt.xlabel('x-pixels (RA)')
plt.ylabel('y-pixels (Dec)')
plt.colorbar()
plt.show()


# In[3]:


#Read Fits
from astropy.io import fits
import numpy as np

def load_fits(filename):
  hdulist = fits.open(filename)
  data = hdulist[0].data

  arg_max = np.argmax(data)  
  max_pos = np.unravel_index(arg_max, data.shape)
  
  return max_pos


# In[4]:


load_fits('image0.fits')


# In[ ]:


#Mean stack


# In[ ]:


#V1
import numpy as np 

def mean_fits(filenames):
    n = len(filenames)
    if n > 0:
        
      hdulist = fits.open('filenames')
      data = hdulist[0].data
        hdulist.close()
      
    
    for i in range (1,n):
        hdulist = fits.open(files[i])
        data += hdulist[0].data
        hdulist.close()
        
      data_mean = data/n
      
      return data_mean


# In[5]:


#V2
from astropy.io import fits
import numpy as np 

def mean_fits(filenames):
    n = len(filenames)
    if n > 0:
        
      hdulist = fits.open(filenames[0])
      data = hdulist[0].data
      hdulist.close()
      
    
      for i in range (1,n):
        hdulist = fits.open(files[i])
        data += hdulist[0].data
        hdulist.close()
        
      data_mean = data/n
      return data_mean


# In[ ]:




