from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# imag.getdata(0)

def detecta_piel():
    capturador = cv2.VideoCapture(0)
    while(True):
        _,cv2_im = capturador.read()
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagbin = imagen_piel(pil_im)
        kernel =construct_kernel()
        opening = cv2.morphologyEx(imagbin.astype("uint8"), cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel, iterations = 1)
        cv2.imshow('frame', dilation*255)# imagbin*255)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
    capturador.release()
    cv2.destroyWindow('frame')

def muestra_camara():
    capturador = cv2.VideoCapture(0)
    while(True):
        _,cv2_im = capturador.read()
        # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', cv2_im)  # imagbin*255)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
    capturador.release()
    cv2.destroyWindow('frame')


def construct_kernel():
    krray = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]]
    return np.asarray(krray, dtype="uint8")


def imagen_cargador(NameImage):
    imag = Image.open(NameImage)
    Red, Green, Blue = imag.split()
    imdata = np.asarray(imag)
    x,y,z = imdata.shape
    binima = np.zeros((x,y))
    Redv = np.asarray(Red)
    Greenv = np.asarray(Green)
    Bluev = np.asarray(Blue)
    for i in range(x):
        for j in range(y):
            if (Redv[i,j] > 95) and (Greenv[i,j] > 40) and (Bluev[i,j]>20) and ((imdata[i,j,:].max() - imdata[i,j,:].min())>15) and ((Redv[i,j]-Greenv[i,j])>15) and (Redv[i,j]>Greenv[i,j]) and (Redv[i,j]>Bluev[i,j]):
                binima[i,j]=1
            else:
                binima[i,j]=0
    mydpi = 100
    fig, axes = plt.subplots(figsize=(y/mydpi,y/mydpi))
    axes.imshow(binima)
    plt.ioff()
    plt.show()
    return binima, fig, imag

def imagen_piel(imagen):
    Red, Green, Blue = imagen.split()
    imdata = np.asarray(imagen)
    x,y,z = imdata.shape
    Redv = np.asarray(Red)
    Greenv = np.asarray(Green)
    Bluev = np.asarray(Blue)
    redv1 = Redv > 95
    greenv1 = Greenv > 40
    bluev1 = Bluev > 20
    imdat1 = (imdata.max(axis=2) - imdata.min(axis=2)) > 15
    re_ge = (Redv-Greenv)>15
    remge = Redv > Greenv
    rembl = Redv > Bluev
    binima = np.logical_and.reduce((redv1, greenv1, bluev1, imdat1,
                                    re_ge, remge, rembl))
    return binima  # .astype("uint8")

def imagen_cargador2(NameImage):
    imag = Image.open(NameImage)
    Red, Green, Blue = imag.split()
    imdata = np.asarray(imag)
    x,y,z = imdata.shape
    binima = np.zeros((x,y))
    Redv = np.asarray(Red)
    Greenv = np.asarray(Green)
    Bluev = np.asarray(Blue)
    redv1 = Redv > 95
    greenv1 = Greenv > 40
    bluev1 = Bluev > 20
    imdat1 = (imdata.max(axis=2) - imdata.min(axis=2)) > 15
    re_ge = (Redv-Greenv)>15
    remge = Redv > Greenv
    rembl = Redv > Bluev
    binima = np.logical_and.reduce((redv1, greenv1, bluev1, imdat1,
                                    re_ge, remge, rembl))
    mydpi = 100
    fig, axes = plt.subplots(1, 2, figsize=(y/mydpi,y/mydpi))
    axes = axes.reshape(-1,)
    axes[0].imshow(imdata)
    axes[0].set_axis_off()
    axes[1].imshow(binima)
    axes[1].set_axis_off()
    plt.ioff()
    plt.show()
    return binima, fig, imag

if __name__ == "__main__":
    if sys.argv[1] == 'p':
        detecta_piel()
    elif sys.argv[1] == 'c':
        muestra_camara()
    elif sys.argv[1] == 'i':
        imagen_cargador2(sys.argv[2])
