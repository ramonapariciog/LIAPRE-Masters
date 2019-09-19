import cv2
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import sys
import time

imagpath = sys.argv[1]
print("Filtrado Gaussiano")
img = cv2.imread(imagpath, 0) # reads image as grayscale
kernel3 = np.array([1,2,1,2,4,2,1,2,1]).reshape(3,-1)*(1/16)
convolucionada = cv2.filter2D(img, -1, cv2.flip(kernel3,-1))
cv2.imshow("Original", img)
cv2.imshow("Filtrada", convolucionada)
cv2.waitKey(5000)
