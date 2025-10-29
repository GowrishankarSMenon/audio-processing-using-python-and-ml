#!/usr/bin/env python
# coding: utf-8

# In[99]:


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np


# In[100]:


debussy_file = "audio/debussy.wav"
duke_file = "audio/duke.wav"
redhot_file = "audio/redhot.wav"


# In[101]:


ipd.Audio(debussy_file)


# In[102]:


ipd.Audio(duke_file)


# In[103]:


ipd.Audio(redhot_file)


# In[104]:


debussy, sr = librosa.load(debussy_file)


# In[105]:


duke, _ = librosa.load(duke_file)


# In[106]:


redhot, _ = librosa.load(redhot_file)


# In[107]:


debussy.size


# In[108]:


sample_duration = 1/sr
print(f"{sample_duration:.6f}")


# In[109]:


plt.figure(figsize=(15, 17))
plt.subplot(3, 1, 1)
librosa.display.waveshow(debussy, alpha=0.5)
plt.title("Debussy")
plt.ylim=(-1, 1)

plt.figure(figsize=(15, 17))
plt.subplot(3, 1, 2)
librosa.display.waveshow(duke, alpha=0.5)
plt.title("Duke")
plt.ylim=(-1, 1)

plt.figure(figsize=(15, 17))
plt.subplot(3, 1, 3)
librosa.display.waveshow(redhot, alpha=0.5)
plt.title("Redhot")
plt.ylim=(-1, 1)

plt.show()


# In[110]:


FRAME_SIZE = 1024 #frames size is the max length that we take to calculate the max amplitude
HOP_SIZE = 512 #hop size is the jumping paramter, this adds overlap to the frame size which is of further use when window funcitons are applied

def amplitude_envelope(signal, frame_size, hop_size):
    amplitude_envelope = []

    for i in range(0, len(signal), hop_size):
        current_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_amplitude_envelope)

    return np.array(amplitude_envelope)


# In[112]:


ae_debussy = amplitude_envelope(debussy, FRAME_SIZE, HOP_SIZE)
print(ae_debussy)


# In[ ]:




