#!/usr/bin/env python3

'''
This is starter code for Lab 7 on Coordinate Frame transforms.

'''

import asyncio
import cozmo
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps, Position, Rotation
import math
import time
import sys

def get_relative_pose(object_pose, reference_frame_pose):

	# get 4x4 matrix for wTr (world-to-robot)
	wTr = np.array(reference_frame_pose.to_matrix().in_column_order).reshape((4,4))
	# get 4x4 matrix for wTo (world-to-object)
	wTo = np.array(object_pose.to_matrix().in_column_order).reshape((4,4))

	# Compute relative transform of cube-to-robot
	relative_transform = np.zeros((4,4))
	relative_transform[:3,:3] = wTr[:3,:3].T @ wTo[:3,:3]
	relative_transform[:,3][:3] = wTr[:3,:3].T @ ((wTo[:,3] - wTr[:,3])[:3])
	relative_transform[3,:] = [0, 0, 0, 1]
	pos = relative_transform[:,3]
	z = R.from_matrix(relative_transform[:3,:3]).as_euler("zxy", degrees = True)[0]

	return Pose(pos[0], pos[1], pos[2], angle_z = degrees(z))

def find_relative_cube_pose(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, prints the pose of the detected cube
	in world coordinate frame and relative to the robot coordinate frame.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None

	while True:
		try:
			cube = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube:
				print("Robot pose: %s" % robot.pose)
				print("Cube pose: %s" % cube.pose)
				print("Cube pose in the robot coordinate frame: %s" % get_relative_pose(cube.pose, robot.pose))
		except asyncio.TimeoutError:
			print("Didn't find a cube")

def move_relative_to_cube(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, when a cube is detected it 
	moves the robot to a given pose relative to the detected cube pose.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None

	while cube is None:
		try:
			cube = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube:
				print("Found a cube, pose in the robot coordinate frame: %s" % get_relative_pose(cube.pose, robot.pose))
		except asyncio.TimeoutError:
			print("Didn't find a cube")

	# Goal tuned based on wanted cube animation
	desired_pose_relative_to_cube = Pose(10, 10, 0, angle_z=degrees(90))

	# Get relative pose between cube and robot
	cTr = get_relative_pose(cube.pose, robot.pose)
	cTr_mat = np.array(cTr.to_matrix().in_column_order).reshape((4,4))

	# Given 4x4 matrix of desired goal in cube frame, convert it to robot frame using the found transform
	goal_pose_in_cube_frame = np.array(desired_pose_relative_to_cube.to_matrix().in_column_order).reshape((4,4))
	goal_pose_in_robot_frame = goal_pose_in_cube_frame @ cTr_mat

	z = R.from_matrix(goal_pose_in_robot_frame[0:3,0:3]).as_euler("zyx", degrees=True)[0]
	pos = goal_pose_in_robot_frame[:,3]

	# Given the pose of the goal, go to it
	cozmo_go_to_pose(robot, pos[1], pos[0], z)
	# Push towards the cube
	cozmo_drive_straight(robot, 30, 50)
	# Lift the cube, rotate 360, and drop
	robot.move_lift(1)
	time.sleep(.7)
	robot.turn_in_place(cozmo.util.degrees(360)).wait_for_completed()
	robot.move_lift(-1)
	time.sleep(1)


# Wrappers for existing Cozmo navigation functions

def cozmo_go_to_pose(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	robot.go_to_pose(Pose(x, y, 0, angle_z=degrees(angle_z)), relative_to_robot=True).wait_for_completed()


def cozmo_drive_straight(robot, dist, speed):
	"""Drives the robot straight.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		dist -- Desired distance of the movement in millimeters
		speed -- Desired speed of the movement in millimeters per second
	"""
	robot.drive_straight(distance_mm(dist), speed_mmps(speed)).wait_for_completed()

def cozmo_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	robot.turn_in_place(degrees(angle), speed=degrees(speed)).wait_for_completed()


if __name__ == '__main__':

	## For step 2
	# cozmo.run_program(find_relative_cube_pose)

	## For step 3
	cozmo.run_program(move_relative_to_cube)


