import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

def show(image):
    cv2.imshow("ventana", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0
    show(labeled_img)
    return labeled_img

ruta = "./"
paths = list(map(lambda x: os.path.join(ruta, x), os.listdir(ruta)))
images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), paths))
show(images[-1])
ima = images[-1]
gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
median_blur_size_kernel = 7
blured = cv2.medianBlur(gray, median_blur_size_kernel)
laplacian_size_kernel = 7
depth = cv2.CV_8U
# laplacian_size_kernel = 7
laplace = cv2.Laplacian(blured, depth, laplacian_size_kernel)
show(laplace)
y, x = np.histogram(laplace.flatten(), bins=250)
plt.figure()
plt.bar(x[1:], y, 0.5)
plt.show()
a, b = cv2.threshold(laplace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show(b)
num_labels, labels_im = cv2.connectedComponents(b)
imshow_components(labels_im)
