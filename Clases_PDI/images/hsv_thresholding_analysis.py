import cv2
import os
import sys
import imutils
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def show(image, title="Imagen"):
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hue_map(ax):
    # creating hue map
    map_hsv = np.tile(np.arange(180), 200).reshape(-1, 180)
    hmap = imshow_components(map_hsv)
    ax.imshow(imutils.opencv2matplotlib(hmap))
    ax.set_title("Hue Map")
    return ax

def show_matr(ax, matr):
    ax.imshow(imutils.opencv2matplotlib(matr))
    ax.set_axis_off()
    ax.set_title("croped")
    return ax

def value_range_matrix(ax, matr):
    im = ax.imshow(matr, cmap=plt.cm.Greys_r)
    # Loop over data dimensions and create text annotations.
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            text = ax.text(j, i, matr[i, j],
                           ha="center", va="center", color="r")
    return ax

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def maskWithCrop(hsv_crop):
    sh = hsv_crop.shape
    layers = cv2.split(hsv_crop)
    q005 = lambda x: np.quantile(x, [0.005])[0]
    q995 = lambda x: np.quantile(x, [0.995])[0]
    mins = list(map(lambda x: int(q005(x.flatten())), layers))
    maxs = list(map(lambda x: int(q995(x.flatten())), layers))
    mi1, mi2, mi3 = mins
    ma1, ma2, ma3 = maxs
    title = "({0}, {1}, {2})".format(mi1, mi2, mi3)
    title += " - ({0}, {1}, {2})".format(ma1, ma2, ma3)
    return [tuple(mins), tuple(maxs)], title

def analysis_fig(hsv_ima, croped, croped_hsv, liminfe, limsupe):
    fig, axes = plt.subplots(4, 2, figsize=(18, 10))
    axes = axes.reshape(-1, )
    axes[0] = hue_map(axes[0])
    axes[1] = show_matr(axes[1], croped)
    for i, l, infe, supe in zip(range(2, 8, 2), range(3), liminfe, limsupe):
        axes[i].hist(hsv_ima[:, :, l].flatten(), bins=100)
        axes[i].axvline(infe, color='g', linestyle='--')
        axes[i].axvline(supe, color='r', linestyle='--')
    for i, l in zip(range(3, 8, 2), range(3)):
        cm = axes[i].imshow(croped_hsv[:, :, l],\
            cmap=plt.cm.Greys_r)
        _ = fig.colorbar(cm, extend='both', ax=axes[i])
    plt.show()

def show_labels(labeled_img, labels_im, masked):
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    axes = axes.reshape(-1, )
    axes[0].imshow(imutils.opencv2matplotlib(labeled_img))
    axes[0].set_title("hue ordered labels")
    cm, axes[1], cmap = discrete_matshow(labels_im, axes[1])
    _ = fig.colorbar(cm, ticks=np.arange(np.min(labels_im),np.max(labels_im)+1),
                     ax=axes[1])
    axes[2].imshow(imutils.opencv2matplotlib(masked))
    axes[2].set_axis_off()
    axes[2].set_title("Masked")
    plt.show()

def discrete_matshow(data, ax):
    #get discrete colormap
    cmap = plt.get_cmap('jet', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = ax.imshow(data, cmap=cmap, vmin = np.min(data)-.5,
                     vmax = np.max(data)+.5)
    #tell the colorbar to tick at integers
    return mat, ax, cmap

def Cleaning_morphological(mask):
    """Close binary objects in binary matrix."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed  # cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_opening)

def show_clusters(clusters_ima):
    fig, ax = plt.subplots()
    cm, ax, cmap = discrete_matshow(clusters_ima, ax)
    ax.set_title("Clustering KMeans")
    _ = fig.colorbar(cm, ticks=np.arange(np.min(clusters_im),np.max(clusters_im)+1),
                     ax=ax)
    plt.show()

if __name__ == "__main__":
    im_path = sys.argv[1]
    ima = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if ima.shape[0] > 1800:
        scale = int(np.ceil(ima.shape[0] / 1080))
        ima = cv2.resize(ima, (ima.shape[1]//scale, ima.shape[0]//scale))
    # show(ima)
    roi = cv2.selectROI(ima)
    blured = cv2.GaussianBlur(ima, (11, 11), cv2.BORDER_DEFAULT)
    croped = blured[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
    show(croped)
    ima_hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    croped_hsv = cv2.cvtColor(croped, cv2.COLOR_BGR2HSV)
    (liminf, limsup), title = maskWithCrop(croped_hsv)
    analysis_fig(ima_hsv, croped, croped_hsv, liminf, limsup)
    # Range threshold
    maskRange = cv2.inRange(ima_hsv, liminf, limsup)
    # morphological operations
    maskRange = Cleaning_morphological(maskRange)
    show(maskRange, title)
    # Labeling
    num_labels, labels_im = cv2.connectedComponents(maskRange)
    labeled_img = imshow_components(labels_im)
    # showing results
    show_labels(labeled_img, labels_im, cv2.bitwise_and(ima, ima, mask=maskRange))
    # clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X=ima_hsv.reshape(-1, 3))
    clusters_im = kmeans.labels_.reshape(ima_hsv.shape[:-1])
    show_clusters(clusters_im)
