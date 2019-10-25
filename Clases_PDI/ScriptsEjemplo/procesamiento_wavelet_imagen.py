import cv2
billete = cv2.imread("Captura7.JPG")
import matplotlib.pyplot as plt
import PIL
PIL.Image.Image.show(billete)
cv2.imshow("billete", billete)
import pywt
pywt.dwt2?
gray_bill = cv2.cvtColor(billete, cv2.COLOR_RGB2GRAY)
cv2.imshow("billete", gray_bill)
cv2.imshow("billete", billete)
pywt.dwt2?
pywt.dwt2(gray_bill, 'daubechies4')
pywt.wavelist()
wt = pywt.dwt2(gray_bill, 'haar')
wt
type(wt)
len(wt)
pywt.dwt2?
len(wt[0])
len(wt[1])
len(wt[1][0])
len(wt[1][1])
len(wt[1][1][1])
gray_bill.shape
wt[]
wt[0]
plt.imshow(wt[0])
plt.show()
plt.imshow(wt[1][0])
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.reshape(-1,)
for ia, (ax, title) in enumerate(zip(axes, ["A", "cH", "cV", "cD"])):
    if ia < 1:
        ax.imshow(wt[0])
    else:
        ax.imshow(wt[1][ia-1])
        ax.title(title)
        ax.set_axis_off()
for ia, (ax, title) in enumerate(zip(axes, ["A", "cH", "cV", "cD"])):
    if ia < 1:
        ax.imshow(wt[0])
    else:
        ax.imshow(wt[1][ia-1])
        ax.set_title(title)
        ax.set_axis_off()
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.reshape(-1,)
for ia, (ax, title) in enumerate(zip(axes, ["A", "cH", "cV", "cD"])):
    if ia < 1:
        ax.imshow(wt[0])
    else:
        ax.imshow(wt[1][ia-1])
        ax.set_title(title)
        ax.set_axis_off()
plt.show()
%hist
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.reshape(-1,)
for ia, (ax, title) in enumerate(zip(axes, ["A", "cH", "cV", "cD"])):
    if ia < 1:
        ax.imshow(wt[0])
        ax.set_title(title)
        ax.set_axis_off()
    else:
        ax.imshow(wt[1][ia-1])
        ax.set_title(title)
        ax.set_axis_off()
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.reshape(-1,)
for ia, (ax, title) in enumerate(zip(axes, ["A", "cH", "cV", "cD"])):
    if ia < 1:
        ax.imshow(wt[0], cmap=plt.cm.gray)
        ax.set_title(title)
        ax.set_axis_off()
    else:
        ax.imshow(wt[1][ia-1], cmap=plt.cm.gray)
        ax.set_title(title)
        ax.set_axis_off()
plt.show()
%hist -f procesamiento_wavelet_imagen.py
