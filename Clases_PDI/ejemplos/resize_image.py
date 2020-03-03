import os
import sys
import argparse
import numpy as np
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InfoManager service.')
    parser.add_argument('--scale', dest='scale', metavar='NNN', type=int,
                        help='Scale factor')
    parser.add_argument('--im', dest='impath', metavar='S', type=str,
                        help='Image path')
    parser.add_argument('--out', dest='imout', metavar='S', type=str,
                        help='Output image path')
    args = parser.parse_args()

    impath = args.impath if args.impath is not None else "test.png"
    scale = args.scale if args.scale is not None else 2
    imoutlist = impath.split(".")
    imout = args.imout + ".%s"%imoutlist[-1] if args.imout is not None else\
        "{0}_resized_{1}x-1x.{2}".format(imoutlist[0], scale, imoutlist[-1])
    imag = cv2.imread(impath, 0)
    h, w = list(map(lambda x: int(x/scale), imag.shape))
    resized = cv2.resize(imag, (w, h))
    cv2.imshow("resized", resized.astype(np.uint8))
    cv2.waitKey(500)
    params = list()
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)
    print("Saving the resized image %s"%imout)
    cv2.imwrite(imout, resized.astype(np.uint8), params)
