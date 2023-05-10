#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.

import sys, cv2, numpy as np, math

#-------------------------------------------------------------------------------
# Main program.

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

# get image from command line    
im_file = sys.argv[1]  
im = cv2.imread(im_file)


#Test to ensure image is read
cv2.imshow("image", im)
cv2.waitKey(0)


#convert to grey
grey = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)
#blur
blur = cv2.GaussianBlur (grey, (5, 5), 0)
#otsu thresholding
t, binary = cv2.threshold (blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#find contours
contours,h = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#find largest contour
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

#Display number of contours and place them on image
#print("Number of contours:" + str(len(contours)))
cv2.drawContours(im, biggest_contour, -1, (0,255,0), 3)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.imshow("image", im)
cv2.waitKey(0)

#find minimum area rotated rectangle	(code used and adapted from https://theailearner.com/tag/cv2-minarearect/)

rect = cv2.minAreaRect(biggest_contour) 	#finds rectangle


#adapted from (https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python)

#angle of rotation and image size
angle = rect[2]
rows, cols = im.shape[0], im.shape[1]
#rotate image
matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
rotate_im = cv2.warpAffine(im,matrix,(cols,rows))
# rotate bounding box
rect_b = (rect[0], rect[1], angle)
box = cv2.boxPoints(rect_b)
pts = np.int0(cv2.transform(np.array([box]), matrix))[0]
pts[pts < 0] = 0

# crop
im_crop = rotate_im[pts[1][1]:pts[0][1],pts[1][0]:pts[2][0]]

#test to display cropped image
cv2.imshow("image", im_crop)
cv2.waitKey(0)

#find red triangle
#change to hsv
crop_hsv = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)

#colour ranges taken from https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/)
#define red colour range (lower boundary)
min_red1 = np.array([0, 100, 20])
max_red1 = np.array([10, 255, 255])
#define red colour range (upper boundary)
min_red2 = np.array([160, 100, 20])
max_red2 = np.array([179, 255, 255])
#create masks
lower_mask = cv2.inRange(crop_hsv, min_red1, max_red1)
upper_mask = cv2.inRange(crop_hsv, min_red2, max_red2)
#combine masks
mask = lower_mask + upper_mask;
#find contour of red triangle
contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(im_crop, contours, 0, (0,255,0), 3)
cv2.imshow("image", im_crop)
cv2.waitKey(0)

for cnt in contours:
    area = cv2.contourArea(cnt)
    # Shortlisting the regions based on there area.
    if area > 1000:
        approx = cv2.approxPolyDP(cnt,
                                  0.1 * cv2.arcLength(cnt, True), True)
        # Checking if the no. of sides of the selected region is 7.
        cv2.drawContours(area, [approx], 0, (0, 0, 255), 5)

arrow_corner = []
for point in approx:
    arrow_corner.append(point[0])


# find the position of red arrow using distance
arrow_corner = np.array(arrow_corner)
arrow_corner2 = arrow_corner.reshape(arrow_corner.shape[0], \
                                     1, arrow_corner.shape[1])
dist = np.sqrt(np.einsum('ijk, ijk->ij', arrow_corner - arrow_corner2, \
                         arrow_corner - arrow_corner2))
dist_sum = np.sum(dist, axis=0)
pos_coord = arrow_corner[dist_sum == np.max(dist_sum), :]
xpos = np.round(pos_coord[0][0] / cols, 3)
ypos = 1 - np.round(pos_coord[0][1] / rows, 3)

# compute the bearing
center_coord = np.mean(arrow_corner[dist_sum != np.max(dist_sum), :], axis=0)
a = [0, -1]
b = pos_coord - center_coord
b = b.reshape(-1)
cosTh = np.dot(a, b)
sinTh = np.cross(a, b)
hdg = np.rad2deg(np.arctan2(sinTh, cosTh))
if hdg < 0:
    hdg = hdg + 360

# Output the position and bearing in the form required by the test harness.
print("POSITION %.3f %.3f" % (xpos, ypos))
print("BEARING %.1f" % hdg)









#cv2.drawContours(im_crop, contours, -1, (0,255,0), 3)

#finds largest contour and finds min enclosing triangle
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
_, triangle = cv2.minEnclosingTriangle(biggest_contour)


#cv2.drawContours(im_crop, triangle, -1, (0,255,0), 3)
#cv2.imshow("image", im_crop)
#cv2.waitKey(0)

#extract points
[[P1x, P1y]] = triangle[0]
[[P2x, P2y]] = triangle[1]
[[P3x, P3y]] = triangle[2]


#find length of sides of triangle (pythagoras)
length1 = math.sqrt((abs(P1x-P2x)**2)+(abs(P1y-P2y)**2))
length2 = math.sqrt((abs(P2x-P3x)**2)+(abs(P2y-P3y)**2))
length3 = math.sqrt((abs(P3x-P1x)**2)+(abs(P3y-P1y)**2))

#find the two sides that are closest in length
val1 = abs(length1 - length2)
val2 = abs(length2 - length3)
val3 = abs(length3 - length1)

point = min(val1, val2, val3)

shape = im_crop.shape

##final point (between 0 and 1)
if point == val1:
	xpos = (1 / shape[1])*P2x 
	ypos = (1 / shape[0])*P2y
	
	#declare point for finding bearing
	point_tip = (P2x, P2y)
	#calculating mid point of shortest side
	if P1x > P3x:
		point_mid = (P1x-((P1x - P3x)/2), P1y-((P1y - P3y)/2))
	else:
		point_mid = (P3x-((P3x - P1x)/2), P3y-((P3y - P1y)/2))
	
if point == val2:
	xpos = (1 / shape[1])*P3x 
	ypos = (1 / shape[0])*P3y
	
	#declare point for finding bearing
	point_tip = (P3x, P3y)
	#calculating mid point of shortest side
	if P1x > P2x:
		point_mid = (P1x-((P1x - P2x)/2), P1y-((P1y - P2y)/2))
	else:
		point_mid = (P2x-((P2x - P1x)/2), P2y-((P2y - P1y)/2))
	
if point == val3:
	xpos = (1 / shape[1])*P1x 
	ypos = (1 / shape[0])*P1y
	
	#declare point for finding bearing
	point_tip = (P1x, P1y)
	#calculating mid point of shortest side
	if P2x > P3x:
		point_mid = (P2x-((P2x - P3x)/2), P2y-((P2y - P3y)/2))
	else:
		point_mid = (P3x-((P3x - P2x)/2), P3y-((P3y - P2y)/2))


##calculating bearing
#simple function using arctan2 to find the angle in radians then convert
#needs middle point of shortest side


#ang1 = numpy.arctan2(*point_mid[::-1])
#ang2 = numpy.arctan2(*point_tip[::-1])
#hdg =  numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))

lat1 = math.radians(point_mid[0])
lat2 = math.radians(point_tip[0])

diffLong = math.radians(point_tip[1]-point_mid[1])
x = math.sin(diffLong)*math.cos(lat2)
y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

bearing = math.atan2(x, y)

bearing = math.degrees(bearing)

hdg = (bearing +360) %360





print ("The filename to work on is %s." % sys.argv[1])


# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

