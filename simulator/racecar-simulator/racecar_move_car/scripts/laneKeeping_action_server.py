#!/usr/bin/env python

import rospy
import actionlib
import threading
from racecar_move_car.msg import  MoveCarGoal, MoveCarAction

# Lock to synchronize the action server with the actual action execution
activity_lock = threading.Lock()
activity_lock.acquire()

#Lane keeping imports
from racecar_control.msg import drive_param

from nav_msgs.msg import Odometry
import math
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import os 

import roslib
roslib.load_manifest('racecar_navigation')
import sys
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import collections
#Lane keeping imports

##__________________________________________________________________________________________##
##__________________________________________________________________________________________##
##__________________________________________________________________________________________##
##                      L A N E K E E P I N G   M O D U L E                                 ##
##__________________________________________________________________________________________##
##__________________________________________________________________________________________##
##__________________________________________________________________________________________##



##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                                 L I N E     C L A S S                                    ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
class Line:
    """
    Class to model a lane-line.
    """

    def __init__(self, buffer_len=10):
        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all rover-centric pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.
        :param new_fit_pixel: new polynomial coefficients (pixel)
        :param new_fit_meter: new polynomial coefficients (meter)
        :param detected: if the Line was detected (True) or inferred (False)
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    def clear_buffer(self):
        self.recent_fits_pixel = []
        self.recent_fits_meter = []

##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                                      G L O B A L S                                       ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
       
debug_lines = False  # Debug : Shows the two lines before and after perspective transform perpendicular to our rover.
verbose = False
scale = 100.0  # pixel/m
ym_per_pix=1/scale
xm_per_pix=1/scale
processed_frames = 0     # counter of frames processed (when processing video) # currently not used
 
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                             I M A G E   F U N C T I O N S                                ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
def color_thresh(img, rgb_thresh=(250.0, 250.0, 250.0)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all thre threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) \
                   & (img[:, :, 2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 255
    # Return the binary image
    return color_select

def birdeye(img):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]
    # img = color_thresh(img, (230, 230, 230))

    scale = 100.0  # pixels per cm
    dst_size_x = 0.4175 * scale
    dst_size_y = 2.5 * scale
    bottom_offset = 1.06 * scale

    src = np.float32([[78, 1024],  # bl
                      [1200, 1024],  # br
                      [703, 855],  # tr
                      [576, 855]])  # tl
    dst = np.float32([[w / 2 - dst_size_x, h - bottom_offset],
                      [w / 2 + dst_size_x, h - bottom_offset],
                      [w / 2 + dst_size_x, h - 2 * dst_size_y - bottom_offset],
                      [w / 2 - dst_size_x, h - 2 * dst_size_y - bottom_offset],
                      ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if (debug_lines):  # Draw vertical line bisecting image to (left|right) before perspective transform
        # (should overlap with the other line drawn before the transform)
        warped = cv2.line(warped, (int(w / 2), h), (int(w / 2), 0), (0, 255, 0), thickness=3, lineType=8, shift=0)

    return warped, M, Minv

def img_preprocess(img, debug_lines):
    h, w = img.shape[:2]

    # Resize image if needed:
    if (w != 1280.0 or h != 1024.0):
        print("WARNING: Width of image = "+str(img.shape[1])+" != 1280 or  Height of image = "+str(img.shape[0])+" !=1024. \n RESIZING!")
        img = cv2.resize(img, (1280, 1024))
        h, w = img.shape[:2]  # should be h=1024, w=1280
    # -----------------------------------------------------------#
    if (debug_lines):  # Draw vertical line bisecting image to (left|right) before perspective transform:
        cv2.line(img, (int(w / 2), h), (int(w / 2), 0), (255, 0, 0), thickness=10, lineType=8, shift=0)
    # -----------------------------------------------------------#
    # Perspective Transform:
    img_birdeye, M, Minv = birdeye(img)
    # -----------------------------------------------------------#
    # Binarization:
    img_birdeye_binary = color_thresh(img_birdeye, (230, 230, 230))  # 1D binary image (to retrieve ypos and xpos)
    img_birdeye = np.dstack((img_birdeye_binary, img_birdeye_binary, img_birdeye_binary))  # 3D binary image (for display)
    # -----------------------------------------------------------#
    ypos, xpos = img_birdeye_binary.nonzero()  # coordinates of the white points after thresholding
    # -----------------------------------------------------------#
    # Change to rover-centric:
    '''  
    Here, the (0,0) of the robot is assumed to be at: (w/2,h)
    So, we need to to get (x-w/2,y-h) for each point (x,y). 
    NB: Rover is not at h-bottom_offset, bottom_offset is the starting point of the shape we are measuring, 
    the white rectangle Nadine put and gave me measurements to..in our case.
    '''
    y_px_centric = -(ypos - h).astype(np.float)
    x_px_centric = (xpos - w / 2).astype(np.float)

    return img_birdeye, img_birdeye_binary, xpos, ypos, x_px_centric, y_px_centric, Minv

def get_left_and_right_peaks(img_birdeye_binary):
    h, w = img_birdeye_binary.shape[:2]
    # Get starting positions of left lane and right lane:
    # Edit the line below if you are getting erroneous start positions. You may want increase (or, more probably, decrease) the summed area.
    histogram = np.sum(img_birdeye_binary[h // 2:h, :], axis=0)  # Note: Only the part till height/2 is summed.
    midpoint = len(histogram) // 2
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    leftx_base = np.argmax(histogram[:midpoint])  # in image coordinates
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # in image coordiantes
    leftx_base_x = leftx_base - w / 2  # in rover-centric coordinates
    rightx_base_x = rightx_base - w / 2  # in rover-centric coordinates
    return leftx_base, rightx_base

##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                          W A Y P O I N T    F U N C T I O N S                            ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##

def get_line_fits_from_sliding_window(birdeye_binary, xm_per_pix, ym_per_pix,leftx_base, rightx_base, xpos, ypos, x_px_centric, y_px_centric,
                                      line_lt, line_rt, n_windows=25, verbose=False):
    """
    Fits left and rigt lane lines by using a sliding window. If it can't, it uses the input lines' last fit.

    :param birdeye_binary: birdeye binary THREE DIMENSIONAL (3 replicate dimensions) image.

    :param xpos: x pixel positions for non-zero pixels in  birdeye_binary
    :param ypos: y pixel positions for non-zero pixels in  birdeye_binary

    :param x_px_centric: x pixel positions for non-zero pixels in  birdeye_binary but robot centric
        x_px_centric = (xpos - w/2 ).astype(np.float)
    :param y_px_centric: y pixel positions for non-zero pixels in  birdeye_binary but robot centric
        y_px_centric=-(ypos - h).astype(np.float)

    :param line_lt: an object of the class Line that is to be or was previously assigned as the left lane line
    :param line_rt: an object of the class Line that is to be or was previously assigned as the right lane line
    :param n_windows: number of windows to split the image height into, when searching
    :param verbose: if True, the output image with rectangles drawn on it is returned (thrid output)


    :return: left_fit_pixel, right_fit_pixel, line_lt-> Updated, line_rt -> Updated, output_img -> if verbose only
    """

    # Set height of windows given number of windows:
    h, w = birdeye_binary.shape[:2]
    window_height = np.int(h / n_windows)

    margin = 25  # width of the windows +/- margin #Note: #Edit this if you find your code is lagging. This means that you are using
    # too many windows for your processor, per frame. Note that, however, more windows generally means more accuracy.
    # Don't forget to change the minipix accordingly. Larger number of windows ->  smaller windows -> less minipix
    minpix = 20  # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    leftx_current = leftx_base  # non rover-centric
    rightx_current = rightx_base  # non rover-centric

    if (verbose):
        out_img = birdeye_binary.copy()

    for win_n in range(n_windows):  # Loop over window numbers
        # Identify this window boundaries in x and y (and right and left)
        # Note: Each iteration, two windows are processed. They have the same height.
        # One on the left line and one on the right line.
        win_y_low = h - (win_n + 1) * window_height  # y_low for both windows
        win_y_high = h - win_n * window_height  # y_high for both windows

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if (verbose):
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((ypos >= win_y_low) & (ypos < win_y_high) & (xpos >= win_xleft_low)
                          & (xpos < win_xleft_high)).nonzero()[0]
        good_right_inds = ((ypos >= win_y_low) & (ypos < win_y_high) & (xpos >= win_xright_low)
                           & (xpos < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(xpos[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(xpos[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    line_lt.all_x, line_lt.all_y = x_px_centric[left_lane_inds], y_px_centric[left_lane_inds]
    line_rt.all_x, line_rt.all_y = x_px_centric[right_lane_inds], y_px_centric[right_lane_inds]

    detected = True
    # The four if/else statements below are error handling that will only blow-up if: first frame and no-lane detected,
    # since there would be no line_xt.last_fit_ANYTHING . A good idea to make the code #Better would be to put
    # the error handling inside a try except that return a straight line in the except statement, but does not place it
    # in the last_fit of any line to avoid assuming the last detected lane was a straight line.

    if not list(line_lt.all_x) or not list(line_lt.all_y):
        # No left-lane pixels this frame: reuse the last good fit, or a neutral straight
        # line on the first frame. Never polyfit an empty vector (it crashed the node).
        left_fit_pixel = line_lt.last_fit_pixel if line_lt.last_fit_pixel is not None else np.array([0.0, 0.0, 0.0])
        left_fit_meter = line_lt.last_fit_meter if line_lt.last_fit_meter is not None else np.array([0.0, 0.0, 0.0])
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel if line_rt.last_fit_pixel is not None else np.array([0.0, 0.0, 0.0])
        right_fit_meter = line_rt.last_fit_meter if line_rt.last_fit_meter is not None else np.array([0.0, 0.0, 0.0])
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)

    if (verbose):
        # Comment this if not plotting:
        left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]
        # ----------------------------------------------------------------------------------------------#
    if (verbose):
        return left_fit_pixel, right_fit_pixel, line_lt, line_rt, out_img
    else:
        return left_fit_pixel, right_fit_pixel, line_lt, line_rt

def get_waypoint_from_fit(y_upfront, left_fit_pixel, right_fit_pixel, ym_per_pix, xm_per_pix, in_meter=False):
    """
    Returns waypoints from the left and right line pixel fit.

    :param y_upfront: distance forward from the robot (in meters) to calculate the horizontal position for.

    :param left_fit_pixel: contains [a,b,c] for left line, assuming ax**2+bx+c is the fitted equation.
    :param right_fit_pixel: contains [a,b,c] for right line, assuming ax**2+bx+c is the fitted equation.

    :param ym_per_pix: meters per pixel on the y-axis. Typically equal to that on the x-axis
    :param xm_per_pix: meters per pixel on the x-axis. Typically equal to that on the y-axis

    :param in_meter: If True, return x,y in meter and Gazebo coordinates. If False, return x,y in pixel and my coordinates.


    :return: left_fit_pixel, right_fit_pixel, line_lt-> Updated, line_rt -> Updated, output_img -> if verbose only
    """

    waypoints_pix_y = y_upfront * 1.0 / ym_per_pix
    waypoints_pix_x_lt = left_fit_pixel[0] * waypoints_pix_y ** 2 + left_fit_pixel[1] * waypoints_pix_y + \
                         left_fit_pixel[2]
    waypoints_pix_x_rt = right_fit_pixel[0] * waypoints_pix_y ** 2 + right_fit_pixel[1] * waypoints_pix_y + \
                         right_fit_pixel[2]
    waypoints_pix_x = (waypoints_pix_x_lt + waypoints_pix_x_rt) / 2

    '''
    Gazebo Coords:            My Coords:
          x^                      y^
           |                       |
           |                       |
     y<----*----> -y       -x<-----*------>x

     * is the car
     y_gazebo=-1*my_x_pix*xm_per_pix #remember to scale  and inverse the sign
     x_gazebo=my_y_pix*ym_per_pix #remember to scale
    '''
    waypoints_m_y = -waypoints_pix_x * xm_per_pix  # Waypoint in meter #Exchange x and y coordiantes for Gazebo
    waypoints_m_x = waypoints_pix_y * ym_per_pix  # Waypoint in meter #Exchange x and y coordiantes for Gazebo

    if (in_meter):
        return (waypoints_m_x, waypoints_m_y)
    else:
        return (waypoints_pix_x, waypoints_pix_y)


##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                           S T A N L E Y    C O N T R O L L E R                           ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
def stanleyController(path_points, velocity, yaw, vehicle_length):
    
    '''
    Applies the lateral controller, Stanley, and returns the required steering angle

    :param path_points: the path points to be followed by the robot (in the robot coordinates frame)
    '''

    # find the path point closest to the vehicle
    min_idx       = 0
    min_dist      = float("inf")

    for i in range(len(path_points)):
        dist = np.linalg.norm(np.array([
                path_points[i][0] - (vehicle_length/2)*np.cos(yaw),
                path_points[i][1] - (vehicle_length/2)*np.sin(yaw)]))
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    # if closest path point is the last one, choose the path point before the last to be able to access min_idx+1 later 
    if min_idx == len(path_points)-1:
        min_idx = min_idx - 1           
    
    # calculate the heading
    psi = -yaw + np.arctan2((path_points[min_idx+1][1]-path_points[min_idx][1]),(path_points[min_idx+1][0]-path_points[min_idx][0]))

    # Logging
    #rospy.loginfo("Yaw:")
    #rospy.loginfo(yaw)

    # calculate an angle proportional to the cross-track error
    phi1 = np.arctan2((path_points[min_idx][1] - (vehicle_length/2)*np.sin(yaw)), path_points[min_idx][0] - (vehicle_length/2)*np.cos(yaw))
    phi2 = np.arctan2((path_points[min_idx+1][1] - path_points[min_idx][1]), (path_points[min_idx+1][0] - path_points[min_idx][0]))

    # make sure it is from -pi to pi
    phi = (np.mod(phi1 + np.pi, 2 * np.pi) - np.pi) - (np.mod(phi2 + np.pi, 2 * np.pi) - np.pi)
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

    # calculate the cross-track error
    err = np.linalg.norm(np.array([
                path_points[min_idx][0] - (vehicle_length/2)*np.cos(yaw),
                path_points[min_idx][1] - (vehicle_length/2)*np.sin(yaw)]))
    err = err * (phi/abs(phi))


    # calculate the required steering angle
    k = 0.005  # tunable gain 1
    ks = 0.1 # tunable gain 2

    err_term = k * (np.arctan2((err), (ks + velocity)))
    err_term = np.mod(err_term + np.pi, 2 * np.pi) - np.pi

    angle = err_term + psi

    angle = np.clip(angle, -0.4189, 0.4189) # 0.4189 radians = 24 degrees because car can only turn 24 degrees max

    return angle

##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##                           L A N E    K E E P I N G    C L A S S                          ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
class LaneKeeping:
    def __init__(self):

        self.active = 0 # boolean to indicate if lane keeping is requested 

        self.velocity =  0 # lane keeping velocity of vehicle (m/s)
        self.vehicle_length = 0.58 # length of vehicles (m) - please adjust this value if another vehicle model is used

        self.yaw = 0 # vehicle's yaw

        self.pub = rospy.Publisher('drive_parameters', drive_param, queue_size=1) # publisher for 'drive_parameters' (speed and steering angle)

        self.line_lt = Line()         # line on the left of the lane
        self.line_rt = Line()         # line on the right of the lane

    def velCallback(self, velMsg):
        '''
        Sets the vehicle's desired velocity (if lane keeping is requested)
        
        :param odomMsg: the float32 message sent through the topic: '/desired_vel'
        '''

        if (self.active == 1):
            self.velocity = velMsg.data


    def odometryCallback(self, odomMsg):
        '''
        Extracts the vehicle's current yaw from the current Odometry message (if lane keeping is requested)
        
        :param odomMsg: the Odometry message sent through the topic: '/vesc/odom'
        '''

        if (self.active == 1):
		# extract the vehicle's current orientation in the form of a quaternion
		orientation_q = odomMsg.pose.pose.orientation
		orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

		# transform from a quaternion to euler angles
		(r, p, y) = euler_from_quaternion (orientation_list)

		# extract the vehicle's yaw
		self.yaw = y


    def imageCallback(self, rosImg):
        '''

        Using Stanley controller (if lane keeping is requested), publishes (on the topic '/drive_parameters') the vehicle's 
        drive_param message (velocity and steering angle) necessary for keeping the lane by 
        means of following the waypoints calculated after applying computer vision techniques 
        on the input image from the vehicle's camera

        :param rosImg: the ROS image obtained from the vehicle's camera through the topic: '/camera/image_raw'
        '''

        if (self.active == 1):
		# change the input ROS image into a CV image to apply the computer vision algorithm on
		bridge = CvBridge()
		cv_img = bridge.imgmsg_to_cv2(rosImg, "mono8") # encoding "mono8" used since the input ROS image is b&w (format L8)
		cv_img = cv_img.astype(np.float64) # change type to float64
		cv_img = np.dstack((cv_img, cv_img, cv_img))  # 3D image

		#cv2.imwrite(r"home/nadine/tmp/cv_img.jpg", cv_img)
		#cv2.imshow("frame",np.array(cv_img, dtype = np.uint8))
		#np.save(r"home/nadine/tmp/testarray.npy", cv_img)
		#h,w=cv_img.shape[:2]

		# preprocess image and get bird's eye view
		img_birdeye,img_birdeye_binary,xpos,ypos,x_px_centric,y_px_centric,Minv=img_preprocess(cv_img,debug_lines)

		# get starting positions of left lane and right lane in non-robot centric pixel coordinates
		leftx_base_x,rightx_base_x= get_left_and_right_peaks(img_birdeye_binary)

		# get line fit coefficients for left and right line from binary thresholded image
		left_fit_pixel, right_fit_pixel, self.line_lt, self.line_rt=get_line_fits_from_sliding_window(img_birdeye,xm_per_pix,ym_per_pix, leftx_base_x, rightx_base_x, xpos, ypos,x_px_centric, y_px_centric,\
		                        self.line_lt, self.line_rt, n_windows=25, verbose=verbose)

		# get way points in meters
		distances_to_get_waypoints_at=np.array([2.06,3.06,4.06]) #in meters : measured front distance from robot

		wp_x,wp_y=get_waypoint_from_fit(distances_to_get_waypoints_at,left_fit_pixel,right_fit_pixel,\
		            ym_per_pix,xm_per_pix,in_meter=True)

		#rospy.loginfo("Y-coordinates of waypoints in the robot-centric coorinates:")
		#rospy.loginfo(wp_y)

		path_points = [((wp_x[0]),(wp_y[0]),(float(0.05))), ((wp_x[1]),(wp_y[1]),(0.05)), ((wp_x[2]),(wp_y[2]),(0.05))] # the three generated waypoints

		# apply the Stanley controller
		angle = stanleyController(path_points, self.velocity, self.yaw, self.vehicle_length)

		# publish the drive_param message
		msg = drive_param()
		msg.velocity = self.velocity
		msg.angle = angle
		self.pub.publish(msg)


##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##            L A N E    K E E P I N G    A C T I O N    S E R V E R    C L A S S           ##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##

class LKActionServer():

    def __init__(self):

        self.a_server = actionlib.SimpleActionServer("move_car/laneKeeping_action_server", MoveCarAction, self.execute_cb, False)
        self.a_server.register_preempt_callback(self.preemptCB)
        self.a_server.start()
        self.lk = LaneKeeping() # object of class LaneKeeping


    def preemptCB(self):
        self.lk.active = 0
        rospy.loginfo("Goal preempted in lane keeping action server")
        
        activity_lock.release()


    # Goal callback function; goal is a MoveCarAction message
    def execute_cb(self, goal):
        
        rospy.loginfo("Received goal in lane keeping action server")

        # Extra check that lane keeping is requested
        if (goal.mcGoal.control_action == 0):

	    self.lk.active = 1

            activity_lock.acquire(False)
            activity_lock.acquire(True)

            self.a_server.set_preempted()

            #rate = rospy.Rate(1)

            #rospy.loginfo("Executing goal in lane keeping action server")

            #while(1):
            #    # check that preempt has not been requested by the client
            #    if self.a_server.is_preempt_requested():
	    #        self.lk.active = 0
            #        rospy.loginfo("Goal preempted in lane keeping action server")
            #        self.a_server.set_preempted() #########send result?
            #        break

            #    rate.sleep()
        else:
            rospy.loginfo("Goal is not lane keeping. Please debug your code!")
            self.a_server.set_aborted()
    

if __name__ == "__main__":
    rospy.init_node("laneKeeping_action_server")

    # create object of class LKActionServer
    lkas = LKActionServer()

    # subscribe to '/vesc/odom'
    rospy.Subscriber('/vesc/odom', Odometry, lkas.lk.odometryCallback, queue_size=1)
    # subscribe to 'camera/image_raw'
    rospy.Subscriber('camera/image_raw', Image, lkas.lk.imageCallback)
    # subscribe to 'move_car/desired_lk_vel'
    rospy.Subscriber('move_car/desired_lk_vel', Float32, lkas.lk.velCallback)

    rospy.spin()
