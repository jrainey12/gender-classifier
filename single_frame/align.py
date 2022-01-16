import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
from os.path import join, basename, exists
import argparse
import os
import imutils
import cython


def main(base_dir):

	#iterate through folders and call align all on each folder
   # genders = glob(join(base_dir,'*'))
   # for gen in genders:
    
    subjects = sorted(glob(join(base_dir,'*')))
    
    for sub in subjects:
        sequences = glob(join(sub,'*'))
        
        for seqs in sequences:
            angles = glob(join(seqs,'*'))

            for angs in angles:
                print (angs)
                align_all(angs)


def align_all(base_dir):
	""" Align silhouettes
	
	Args:
		base_dir (str): directory of silhouette images
		
	Returns:
		list: List of aligned images
	"""
	images = []
	final_images= []
	#out_dir = "home/james/smplify_complete/smplify_GEI/results"
		
	image_paths = sorted(glob(join(base_dir, '*.png')))
	if not image_paths == []:
		for i_path in image_paths:	
		#	images.append(align_images(base_dir, i_path))
			images.append(align_images(base_dir, i_path, pad_image(i_path)))
				
	
	final_images = [x for x in images if x is not None]	

			
	return final_images
			

def convert_to_BW(image):
#	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
#	cv2.imshow("Before",image)
#	cv2.waitKey(0)
	
	thresh = 254
	im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
	im_wb =  cv2.bitwise_not(im_bw)
#	cv2.imshow("after", im_bw)
#	cv2.waitKey(0)
	
	return im_wb 

def align_images(base_path, image_path, image):
	""" Individually crop and align each image
	
	Args:
		base_path (str): Directory of unaligned images
		image_path (str): Directory of single image
		
	Returns:
		list: Aligned images
	
	"""
	
	#read in image
	#img = cv2.imread(image_path,0)
	img = image
	
	height = img.shape[0]
	width = img.shape[1]
	
	#Find top pixel
	for y in range(0, height - 1):
		for x in range(0, width - 1):
			if img[y,x] == 255:
				top = [x,y]
				#print top
				break
		else:
			continue 
		break  


	#Find bottom pixel
	for y1 in range(height - 1, 0,-1):
		for x1 in range(width-1 ,0,-1):
			#print (y1,x1)
			if img[y1,x1] == 255:
				bottom = [x1,y1]
				#print bottom
				break
		else:
			continue 
		break  

	#calculate height of silhouette
	height = bottom[1] - top[1] 
		
	
	#Find center of gravity of the silhouette
	contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv2() else contours[1]
	for c in contours:
		M = cv2.moments(c)
		
		if M["m00"] == 0:
			print (" incomplete contour - " + image_path)
			continue
			
		else:	
			X = int(M["m10"] / M["m00"])
			Y = int(M["m01"] / M["m00"])
	
	middle = (X,Y)
	
	#Draw bounding box around silhouette 
	
	box_width = (11 * height) / 16
	#print "BOX WIDTH/2 : " + str(box_width/2)
	
	#~ diff = 0
	#~ pad_left = False
	#~ pad_right = False
	
	box_right = middle[0] + (box_width/2)
	
	
	if box_right > img.shape[1]:
		print (box_right)
		print ("ERROR")
		return
		
	#possible solution to incorrect alignment one extreme right images
	#Doesnt completely work, causes images to be stretched horizontally
	#if box_right > img.shape[1]:
	#	right_width = img.shape[1] - middle[0]
		#print "RIGHT WIDTH: " + str(right_width)
	#	if right_width < box_width/2:
	#		box_width = right_width*2
	#	diff = box_width/2 - right_width
	#	pad_right = True
		#print " DIFFERENCE RIGHT: " + str(diffr)
	#	box_right = img.shape[1]
	
	
	box_left = middle[0] - (box_width/2)
	
	if box_left < 0:
		print (box_left)
		print ("ERROR")
		return
	#possible solution to incorrect alignment one extreme left images
	#Doesnt completely work, causes images to be stretched horizontally
	#~ if box_left < 0:
		#~ #box_left = 0
		#~ diff = 0 - box_left
		#~ print "DIFF: " +str(diff) 
		#~ pad_left = True 
		#~ box_left = 0
		#print " DIFFERENCE LEFT: " + str(diff)
		#left_width = middle[0] - box_left
		#if left_width == middle[0]:
		#	box_right = middle[0] + left_width
	
		
	#Possibly pad each side with black pixels to make up the width again
	
	
	box_top = top[1]

	box_bottom = bottom[1]

#DOESNT WORK
	#~ if pad_left:
		#~ img = cv2.copyMakeBorder(img,0,0,diff,0,cv2.BORDER_CONSTANT, value=[000])
		#~ pad_left = False
		#~ print "padding left"
	#~ if pad_right:	
		#~ img = cv2.copyMakeBorder(img,0,0,0,diff,cv2.BORDER_CONSTANT, value=[000])
		#~ pad_right = False
		#~ print "padding right"
	
	#Crop the image
	crop_img = img[box_top:box_bottom, box_left:box_right]
	
	
		
	#Resize the image
	resized_image = cv2.resize(crop_img, (88, 128))
	
		
	
	#USED TO OUTPUT ALIGNED IMAGES TO FILE
	outpath = join(base_path, 'aligned')
	if not exists(outpath):
		os.mkdir(outpath)
   	#print (outpath)
	#cv2.imwrite(join(out_dir, basename(image_path)), resized_image)
	cv2.imwrite(join(outpath, basename(image_path)), resized_image)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()			
	
	
	return resized_image
		
		
		
def pad_image(img_path):
	""" Pad an image at the left and right to space for all silhouettes to be cropped properly
	
	Args:
		img_path: path to image to be padded
	
	Returns:
		padded image
	"""
	img = cv2.imread(img_path, 0)
	
	#print (len(img.shape))
	
	#img = convert_to_BW(img)

	padded_img = cv2.copyMakeBorder(img,0,0,50,50,cv2.BORDER_CONSTANT, value=[000])
	
	#cv2.imshow('padded',padded_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	return padded_img
		
if __name__ == '__main__':
	
    parser = argparse.ArgumentParser(description='Convert joint output from Deepcut to be suitable for SMPLify')
	 
    parser.add_argument(
        'base_dir',
        default='/home/james/',
        nargs='?',
        help="Directory that contains the images to be aligned")   
   
    parser.add_argument(
        '--out_dir',
        default='/home/james/smplify_complete/Datasets/silhouettes',
        nargs='?',
        help="Directory to save sils to.")       
	
    args = parser.parse_args()
	#align_all(args.base_dir, args.out_dir)
    main(args.base_dir)
