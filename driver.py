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

U,S,V = functions.singular_value_decomposition(hosico, rank=70)
hosico_compressed = U@S@V
fig,ax=plt.subplots()
ax.imshow(hosico_compressed)
plt.show()