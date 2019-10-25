import numpy as np
import matplotlib.pyplot as plt
import cv2

drake = "111100001010010000110010000000100000101001\
000001100101100111100000110000110100000000100000100\
001000010001010100001000000000000000000010001000000\
000010110000000000000000000100011101101011010100000\
000000000000001001000011101010101000000000101010101\
000000000111010101011101011000000010000000000000000\
010000000000000100010011111100000111010000010110000\
011100000001000000000100000000100000001111100000010\
110001011101000000011001011111010111110001001111100\
100000000000111110000001011000111111100000100000110\
00001100001000011000000011000101001000111100101111"
drake = np.array([c for c in drake], dtype=int)
mensaje = drake.reshape((29,19))
plt.imshow(mensaje, cmap=plt.cm.gray)
plt.axis("off")
# plt.savefig("drake.png", bbox_inches='tight')
cv2.imwrite("drake.png", mensaje*255)
#%%--------------------------------------------------------
arecibo = "00000010101010000000000001010000010100000001001\
0001000100010010110010101010101010101001001000000000000000\
000000000000000000000011000000000 000000000011010000000000\
000000000110100000000000000000010101000000000 000000000111\
110000000000000000000000000000000011000011100011000011000 \
1000000000000011001000011010001100011000011010111110111110\
11111011111 0000000000000000000000000010000000000000000010\
00000000000000000000000 0000100000000000000000111111000000\
00000001111100000000000000000000000 1100001100001110001100\
01000000010000000001000011010000110001110011010 1111101111\
1011111011111000000000000000000000000001000000110000000001\
0 00000000001100000000000000010000011000000000011111100000\
1100000011111 00000000001100000000000001000000001000000001\
0000010000001100000001000 00001100001100000010000000000110\
0010000110000000000000001100110000000 00000011000100001100\
0000000110000110000001000000010000001000000001000 00100000\
0011000000001000100000000110000000010001000000000100000001\
000 001000000010000000100000001000000000000110000000001100\
000000110000000 001000111010110000000000010000000100000000\
000000100000111110000000000 001000010111010010110110000001\
001110010011111110111000011100000110111 000000000101000001\
110110010000001010000011111100100000010100000110000 001000\
0011011000000000000000000000000000000000001110000010000000\
00000 0011101010001010101010100111000000000101010100000000\
00000000101000000 0000000011111000000000000000011111111100\
00000000001110000000111000000 0001100000000000110000000110\
10000000001011000001100110000000110011000 0100010100000101\
00010000100010010001001000100000000100010100010000000 0000\
0100001000010000000000001000000000100000000000000100101000\
0000000 01111001111101001111000"
arecibo = np.array([c for c in arecibo if c.isnumeric()], dtype=int)
mensaje_arecibo = arecibo.reshape((73,23))
plt.imshow(mensaje_arecibo, cmap=plt.cm.gray)
plt.axis("off")
# plt.savefig("arecibo.png", bbox_inches='tight')
cv2.imwrite("arecibo.png", mensaje_arecibo*255)
