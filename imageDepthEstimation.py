import cv2
import argparse
import tensorflow as tf
import numpy as np
import urllib
import time

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img

num = 377

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--left', dest='left',
                        help='Stereo pathL',
                        default='depth.png', type=str)
    parser.add_argument('--right', dest='right',
                    help='Stereo pathR',
                    default='depth.png', type=str)
   
    args = parser.parse_args()
    return args

def imageDepthEstimation(left, right, num):

	# Select model type
	# model_type = ModelType.middlebury
	model_type = ModelType.flyingthings
	# model_type = ModelType.eth3d

	if model_type == ModelType.middlebury:
		# model_path = "models/middlebury_d160.pb"
		# model_path = "models/middlebury_d288.pb"
		model_path = "models/middlebury_d400.pb"
	elif model_type == ModelType.flyingthings:
		model_path = "models/flyingthings_finalpass_xl.pb"
		# model_path = "models/flyingthings_cleanpass_xl.pb"
	elif model_type == ModelType.eth3d:
		model_path = "models/eth3d.pb"

	# < ====== Depend on user's camera configuration ===== >
	# camera_config = CameraConfig(0.06, 35) # (baseline(m), focal lenth(pixel))
	baseline = 0.15
	focal_length = 2487
	camera_config = CameraConfig(baseline, focal_length) # (baseline(m), focal lenth(pixel))
	max_distance = 10

	# Initialize model
	hitnet_depth = HitNet(model_path, model_type, camera_config)
	print('*******model loaded*******')
	start_time = time.time()
	# Load images
	left_img = cv2.imread(left, cv2.IMREAD_UNCHANGED)
	right_img = cv2.imread(right, cv2.IMREAD_UNCHANGED)

	# # resize image
	# left_img = cv2.resize(left_img,dsize=(int(left_img.shape[1]/2), int(left_img.shape[0]/2)))
	# right_img = cv2.resize(right_img,dsize=(int(right_img.shape[1]/2), int(right_img.shape[0]/2)))

	left_img = cv2.resize(left_img,dsize=(960,540))
	right_img = cv2.resize(right_img,dsize=(960,540))

	# # if you wanna crop images...
	# h = left_img.shape[0]
	# w = left_img.shape[1]
	# left_img_crop = left_img[int(h/2):h, :]
	# right_img_crop = right_img[int(h/2):h, :]

	#left_img = cv2.imdecode(np.asarray(bytearray(left_img), dtype=np.uint8), -1)
	#right_img = cv2.imdecode(np.asarray(bytearray(right_img), dtype=np.uint8), -1)
	#left_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	#right_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	# Estimate the depth
	disparity_map = hitnet_depth(left_img, right_img)
	depth_map = hitnet_depth.get_depth()
	# print(depth_map)
	# print(np.max(depth_map))

	color_disparity = draw_disparity(disparity_map)
	color_depth = draw_depth(depth_map, max_distance)
	# color_depth.astype(np.uint16)
	#cobined_image = np.hstack((left_img, right_img, color_disparity))
	start_time = time.time()
	cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated depth", color_depth)
	cv2.waitKey(0)

	if model_path.find("d160") != -1:
		namef="Depth_Image/middlebury_d160_seg"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png"
	elif model_path.find("d288") != -1:
		namef="Depth_Image/middlebury_d288_seg"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png"
	elif model_path.find("d400") != -1:
		namef="Depth_Image/middlebury_d400_seg_"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png"
	elif model_path.find("final") != -1:
		namef="Depth_Image/flythings_finalpass_seg_"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png"
	elif model_path.find("clean") != -1:
		namef="Depth_Image/flythings_cleanpass_seg"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png"
	elif model_path.find("eth") != -1:
		namef="Depth_Image/eth3d_seg_"+str(num)+"_"+str(max_distance)+"_"+str(baseline)+".png" 

	cv2.imwrite(namef, color_depth)
	end_time = time.time()
	print('depth image saved!')
	print('HITNet time consumed : ', end_time-start_time)
	#print(namef)

	return namef, max_distance

if __name__ == '__main__':

    # args = parse_args()
	
	# ********** raw stereo image : .jpeg **********
	# ********** segmented stereo image : .jpg **********
	left_path = "/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/left/seg_left_real_e50_conf75_"+str(num)+".jpg"
	right_path = "/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/right/seg_right_real_e50_conf75_"+str(num)+".jpg"
	imageDepthEstimation(left_path, right_path, num)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()