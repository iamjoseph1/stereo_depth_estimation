import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--StereoPathL', dest='StereoPathL',
                        help='Stereo pathL',
                        default='depth.png', type=str)
    parser.add_argument('--StereoPathR', dest='StereoPathR',
                    help='Stereo pathR',
                    default='depth.png', type=str)
   
    args = parser.parse_args()
    return args

def nothing(x):
    pass

def stereo2depth_SBM(StereoPathL, StereoPathR): 
    imgL = cv2.imread(StereoPathL, 0)
    imgR = cv2.imread(StereoPathR, 0)

    # imgL = cv2.cvtColor(imgL, cv2.COLOR_BGRA2GRAY)
    # imgR = cv2.cvtColor(imgR, cv2.COLOR_BGRA2GRAY)
    #print("Size of imgL : ",imgL.shape)
    #print("Size of imgR : ",imgR.shape)

    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)

    fx = 942.8        # lense focal length
    baseline = 54.8   # distance in mm between the two cameras
    units = 0.512     # depth units, adjusted for the output to fit in one byte


    cv2.createTrackbar('numDisparities','disp',1,17,nothing)
    cv2.createTrackbar('blockSize','disp',5,50,nothing)
    cv2.createTrackbar('preFilterType','disp',1,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
    cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv2.createTrackbar('speckleRange','disp',0,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv2.createTrackbar('minDisparity','disp',5,25,nothing)
    
    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()
    #stereo = cv.StereoBM.create(numDisparities=32, blockSize=15)

    while True:

        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        disparity = stereo.compute(imgL,imgR)

        # Converting to float32 
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        # Displaying the disparity map
        cv2.imshow("disp",disparity)

        # Close window using esc key
        if cv2.waitKey(1) == 27:
            cv2.destroyWindow('disp') 
            break

    print("parameters : ", numDisparities, blockSize, preFilterType, preFilterSize, preFilterCap, textureThreshold, uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff, minDisparity)
    valid_pixels = disparity > 0

    # calculate depth data
    depth = np.zeros(shape=imgL.shape).astype("uint16")
    depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

    ## visualize depth data
    # depth = cv2.equalizeHist(depth)
    # colorized_depth = np.zeros((imgL.shape[0], imgL.shape[1], 3), dtype="uint16")
    # temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    # colorized_depth[valid_pixels] = temp[valid_pixels]
    # plt.imshow(colorized_depth)
    # plt.show()

    timestamp = time.time()
    #depthf.write(repr(timestamp)+" Depth/"+repr(timestamp)+".png\n")
    depth.astype(np.uint16)
    print(depth.dtype)
    namef="Depth/"+repr(timestamp)+".png"    
    cv2.imwrite(namef,depth)
    # load_depth_image = cv2.imread(namef)
    # cv2.imshow("load_depth_image", load_depth_image)
    # print(load_depth_image.dtype)

    return namef

if __name__ == '__main__':

    args = parse_args()
    stereo2depth_SBM(args.StereoPathL, args.StereoPathR)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
