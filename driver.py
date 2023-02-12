import numpy as np
import functions
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('hosicoBowl.PNG')
greyscale= img.convert('L')
hosico = np.array(greyscale)
fig,ax=plt.subplots()
ax.imshow(hosico)
plt.show()

U,S,V = functions.svd(hosico)
hosico_compressed = U@np.diag(S)@V.T
fig,ax=plt.subplots()
ax.imshow(hosico_compressed)
plt.show()