run alien_msg.py
mensaje.shape
mensaje.shape*10
widht, height = mensaje.shape
width, height = mensaje.shape
np.array([[width*10, 0, 0], [0, height*10, 1]], dtype=float32)
np.array([[width*10, 0, 0], [0, height*10, 1]], dtype=np.float32)
T = np.array([[width*10, 0, 0], [0, height*10, 1]], dtype=np.float32)
cv2.warpAffine?
cv2.warpAffine(mensaje*255, T)
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T, (height*10, width*10))
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
mensaje
mensaje*255
cv2.warpAffine(T, mensaje*255, (width*10, height*10))
T
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T, (height*10, widht*10))
cv2.warpAffine(mensaje, T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T.T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
T
T = np.array([[width*10, 0, 0], [0, height*10, 0]], dtype=np.float32)
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
T
cv2.warpAffine(mensaje*255, T, (width*10, height*10))
cv2.warpAffine(mensaje*255, T, (width*20, height*20))
T = np.array([[width*10, 0, 0], [0, height*10, 0],[0, 0, 1]], dtype=np.float32)
T
cv2.warpAffine(mensaje*255, T, (width*20, height*20))
T = np.array([[20, 0, 0], [0, 20, 0],[0, 0, 1]], dtype=np.float32)
cv2.warpAffine(mensaje*255, T, (width*20, height*20))
cv2.warpAffine(mensaje*255, T[:2, :], (width*20, height*20))
cv2.warpAffine(mensaje*255, T[:2, :], (height*20, width*20))
cv2.warpAffine(mensaje*255, T[:2, :], (mensaje.shape[1]*20, mensaje.shape[0]*20))
T
def warpScale(im, scale):
    affine = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ], np.float32)
    return cv2.warpAffine(im, affine[:2, :], (int(im.shape[1]*scale), int(im.shape[0]*scale)))
im2 = warpScale(mensaje, 10)
im2 = warpScale(mensaje, 10)
im2 = warpScale(mensaje, scale=20)
def warpScale(im, scale):
    affine = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ], np.float32)
    return cv2.warpAffine(im, affine[:2, :], (int(im.shape[1]*scale), int(im.shape[0]*scale)))
im2 = warpScale(mensaje, scale=20)
resized = cv2.resize(mensaje*255, (mensaje.shape[1]*10, mensaje.shape[0]*10))
resized = cv2.resize(mensaje, (mensaje.shape[1]*10, mensaje.shape[0]*10))
resized = cv2.resize(mensaje, (200, 200))
resized = cv2.resize(mensaje.astype(np.float32), (200, 200))
im2 = warpScale(mensaje.astype(np.float32)*255, scale=20)
cv2.imwrite("Drake.jpg", im2)
im3 = warpScale(mensaje_arecibo.astype(np.float32)*255, scale=20)
cv2.imwrite("Arecibo.jpg", im3)
%hist
%hist -f opencv_alienmsg.py
