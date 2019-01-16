'''python3 audio2nsgt.py -i dataset_audio/Claps/ -o results -d 1'''

import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0,1,1000)
plt.plot(n,np.sin(100*n))  # on utilise la fonction sinus de Numpy
plt.ylabel('fonction sinus')
plt.xlabel("l'axe des abcisses")
plt.show()

np.load("results/sin_F440.npy")
a = np.load("results/sin_F440.npy")
print(np.shape(a))
plt.imshow(np.abs(np.load('results/sin_F440.npy')).transpose(),origin='lower'); plt.show()
