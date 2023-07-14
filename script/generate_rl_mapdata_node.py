#! /usr/bin/env python

import rospy
import sys
import os
import random
import time
import numpy as np
import tf2_ros
import tf
from gazebo_msgs.srv import DeleteModel, SpawnModel
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from geometry_msgs.msg import *
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import pdb
from scipy.io import savemat
import cv2

MAX_NUM_DATA = 1000

# class metadata():
#     def __init__(self, gridmap, ):
#         self.field1 = field1
#         self.field2 = field2
#         self.field3 = field3

def del_model( strModelName ):
    """ Remove the model with 'modelName' from the Gazebo scene """
    # delete_model : gazebo_msgs/DeleteModel
    rospy.wait_for_service('gazebo/delete_model')
    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel) # Handle to model spawner
    # rospy.wait_for_service('gazebo/delete_model') # Wait for the model loader to be ready
    # FREEZES EITHER WAY
    del_model_prox(strModelName) # Remove from Gazebo


def spawn_model(strModelName, rx, ry):
    rospy.wait_for_service('gazebo/spawn_urdf_model')
    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
    with open("/home/hankm/python_ws/neuro_frontier_detection/model/turtlebot3_description.urdf") as f:
        robot_urdf = f.read()
    orient = Quaternion(x=0, y=0, z=0, w=1)
    rpose = Pose(Point(x=rx, y=ry, z=0), orient)
    spawn_model_prox(strModelName, robot_urdf, "", rpose, "map")

def set_model_state(strModelName, wx, wy):
    state_msg = ModelState()
    state_msg.model_name = strModelName
    state_msg.pose.position.x = wx
    state_msg.pose.position.y = wy
    state_msg.pose.position.z = 0.0
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 1
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )
    except rospy.ServiceException:
        print("Service call failed")


def reset_octomap():
    reset_map = rospy.ServiceProxy('octomap/_server/reset', Empty)

def reset_world():
    reset_sim = rospy.ServiceProxy('gazebo/reset_world', Empty)

def reset_gazebo():
    reset_sim = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def world2grid(gmmsg, wx, wy):
    gx = (wx - gmmsg.info.origin.position.x) / gmmsg.info.resolution
    gy = (wy - gmmsg.info.origin.position.y) / gmmsg.info.resolution
    return int(gx), int(gy)


def grid2world(gmmsg, gx, gy):
    wx = (gx * gmmsg.info.resolution) + gmmsg.info.origin.position.x
    wy = (gy * gmmsg.info.resolution) + gmmsg.info.origin.position.y
    return (wx, wy)


def gridmap2img(gridmap):
    unkn = (gridmap == -1).astype(int)*127
    free = (gridmap == 0).astype(int)*0
    occu = (gridmap == 100).astype(int)*255
    return (unkn + occu + free)


def identify_free_cells(gridmap):
    # identify free region
    free_cell_loc = np.asarray(np.where(gridmap == 0)) # be careful [rows ; columns] i.e) [y, x]
    #free_cell_loc = np.flipud(free_cell_loc)
    return free_cell_loc

def savemap(resdir, gridmap, dataidx=0):
    gmapimg = gridmap2img(gridmap)
    (height, width) = gmapimg.shape
    imgname = '%s/map%05d.pgm' % (resdir, dataidx)
    cv2.imwrite(imgname, gmapimg)
    free_cell_loc = identify_free_cells(gridmap)
    fname_frees = '%s/frees%05d.csv' % (resdir, dataidx)
    np.savetxt(fname_frees, free_cell_loc, delimiter=",")


def savemetadata(fname, dataidx=0,gx_prev=0,gy_prev=0,gx_next=0,gy_next=0,reward=0):
    # save metadata
    f = open(fname, 'w')
    f.write('%d %d %d %d %d' % (gx_prev, gy_prev, gx_next, gy_next, reward))
    f.close()

# def savemapdata(filename,gridmap,dataidx=0,gx_prev=0,gy_prev=0,gx_next=0,gy_next=0,reward=0):
#     fname = '%s%05d.dat' % (filename, dataidx)
#     # save img and data
#     gmapimg = gridmap2img(gridmap)
#     (height, width) = gmapimg.shape
#     imgname = '%s_map%05d.pgm' % (filename, dataidx)
#     cv2.imwrite(imgname, gmapimg)
#     f = open(fname, 'w')
#     f.write('%d %d %d %d %d %d %d' % (gx_prev, gy_prev, gx_next, gy_next, reward, height, width))
#     f.close()
#     free_cell_loc = identify_free_cells(gridmap)
#     fname_frees = 'frees%05d.csv' % dataidx
#     np.savetxt(fname_frees, free_cell_loc, delimiter=",")


def main(argv):
    workdir = os.getcwd()
    resdir  = '/home/hankm/results/cnn_map_reward'
    rospy.init_node('gen_mapdata_node')
    #rospy.subscriber("/move_base/global_costmap/costmap", OccupancyGrid, costmapCallback)

    #listener = tf.TransformListener()
    # id init robot pose
    # rate = rospy.Rate(1.0)
    # try:
    #     (trans, rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
    #     rospy.loginfo('Trans: ' + str(trans))
    # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #     print("cannot find tform of /map to /base_link")
    # rate.sleep()

    dataidx = 0
    #tfBuffer = tf2_ros.Buffer()
    #listener = tf2_ros.TransformListener(tfBuffer)
    listener = tf.TransformListener()

    # dummy repositioning ... b/c /odom changes its name to /turtlebot3/odom after the repositioning

    # odommsg = rospy.wait_for_message("/odom", Odometry, timeout=None)
    # (wx_init, wy_init) = (odommsg.pose.pose.position.x, odommsg.pose.pose.position.y)
    # del_model(str('turtlebot3'))
    # time.sleep(0.5)
    # spawn_model(str('turtlebot3'), wx_init, wy_init)
    # time.sleep(0.8)
    # print("%f %f"%(odommsg.pose.pose.position.x, odommsg.pose.pose.position.y))
    # gmmsg_prev = rospy.wait_for_message("/map", OccupancyGrid, timeout=None)
    # (wx_prev, wy_prev) = (odommsg.pose.pose.position.x, odommsg.pose.pose.position.y)
    # (gx_prev, gy_prev) = world2grid(gmmsg_prev, wx_prev, wy_prev)
    # gridmap = np.array(gmmsg_prev.data).reshape((gmmsg_prev.info.height, gmmsg_prev.info.width))
    # explored_map = (gridmap != -1).astype(int)
    # previous_explored_area_size = explored_map.sum() * gmmsg_prev.info.resolution * gmmsg_prev.info.resolution

    # save img and meta data of the init map
    #savemetadata(gridmap, dataidx=0, gx_prev=gx_prev, gy_prev=gy_prev, gx_next=gx_prev, gy_next=gy_prev, reward=0)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown(): #dataidx < 10000:
        listener = tf.TransformListener()
        print("generating data: %d" % dataidx)

    # load gridmap b4 moving the robot
        print("req /turtlebot3/odom msg\n")
        # try:
        #     trans = tfBuffer.lookup_transform('map', 'base_footprint', rospy.Time())
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     #print('tf exception: lookup %d connection %d extrap %d' % (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException))
        #     continue
        #     rate.sleep()
        t = rospy.Time(0)
        listener.waitForTransform('map', 'base_link', t, rospy.Duration(0))
        (trans, rot) = listener.lookupTransform("map", "base_link", t)

        print("got the tf msg b4 moving (%f %f)\n" %  (trans[0], trans[1]) )#(trans.transform.translation.x, trans.transform.translation.y))
        (wx_prev, wy_prev) = (trans[0], trans[1]) #(trans.transform.translation.x, trans.transform.translation.y)

        # odommsg = rospy.wait_for_message("/odom", Odometry, timeout=None)
        # (wx_prev, wy_prev) = (odommsg.pose.pose.position.x, odommsg.pose.pose.position.y)

        gmmsg_prev = rospy.wait_for_message("/map", OccupancyGrid, timeout=None)
        print("got /map msg\n")

        (gx_prev, gy_prev) = world2grid(gmmsg_prev, wx_prev, wy_prev)
        gridmap = np.array(gmmsg_prev.data).reshape((gmmsg_prev.info.height, gmmsg_prev.info.width))
        explored_map = (gridmap != -1).astype(int)
        previous_explored_area_size = explored_map.sum() * gmmsg_prev.info.resolution * gmmsg_prev.info.resolution
    ##########################################################################
    # find a random move position, then move the robot to there.
    ##########################################################################
    # identify free region
        free_cell_loc = identify_free_cells(gridmap)
        (dummy, num_free_cell) = free_cell_loc.shape

    # draw a random sample from free region
        sample_idx = random.sample(range(0, num_free_cell-1), 1)[0]
        print("sample idx/ tot pts (%d / %d) " % (sample_idx, num_free_cell-1))
        gy_next = free_cell_loc[0, sample_idx] # row idx
        gx_next = free_cell_loc[1, sample_idx] # column idx

        #pdb.set_trace()
        # remove / respawn robot to the random free cell
        (wx_next, wy_next) = grid2world(gmmsg_prev, gx_next, gy_next)

        # save data b4 we make the movement
        #imgname = '%s/map_before_moving%05d.pgm' % (resdir, dataidx)
        savemap(resdir, gridmap, dataidx)
        fname = '%s/metadata_before_moving%05d.dat' % (resdir, dataidx)
        savemetadata(fname, dataidx, gx_prev, gy_prev, gx_next, gy_next, 0)
        print("finished saving data b4 moving the robot \n")
    # re-positioning the robot
        del_model(str('turtlebot3'))
        print("robot removed \t")
        time.sleep(0.5)
        spawn_model(str('turtlebot3'), wx_next, wy_next)
        print("then it is respawned \n")
        #set_model_state('turtlebot3', wx_next, wy_next)
        time.sleep(0.5)

    # load new map w/ updated/new scanned region
        print("req /map msg\n")
        gmmsg_next = rospy.wait_for_message("/map", OccupancyGrid, timeout=None)
        print("got /map msg\n")
        gridmap = np.array(gmmsg_next.data).reshape((gmmsg_next.info.height, gmmsg_next.info.width))
        explored_map = (gridmap != -1).astype(int)
        explored_area_size = explored_map.sum() * gmmsg_next.info.resolution * gmmsg_next.info.resolution
        reward = explored_area_size - previous_explored_area_size
        reward = max(reward, 0) # This wont happen but just in case

        print("reading tf to base_footprint \n")
        #try:
        # tfBuffer.wait
        # trans = tfBuffer.lookup_transform('map', 'base_footprint', rospy.Time())
        #except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #    continue
        #    rate.sleep()

        t = rospy.Time(0)
        listener.waitForTransform('map', 'base_link', t, rospy.Duration(0))
        (trans, rot) = listener.lookupTransform("map", "base_link", t)

        print("got the moved new tf msg (%f %f)\n"%  (trans[0], trans[1]) )#(trans.transform.translation.x, trans.transform.translation.y))
        (wx, wy) = (trans[0], trans[1]) #(trans.transform.translation.x, trans.transform.translation.y)

        # odommsg = rospy.wait_for_message("/odom", Odometry, timeout=None)
        # (wx, wy) = (odommsg.pose.pose.position.x, odommsg.pose.pose.position.y)

        (gx, gy) = world2grid(gmmsg_next, wx, wy)
    # save the map after moving the robot and the metadata
        #imgname = '%s/map_after_moving%05d.pgm' % (resdir, dataidx)
        savemap(resdir, gridmap, dataidx)
        fname = '%s/metadata_after_moving%05d.dat' % (resdir, dataidx)
        savemetadata(fname, dataidx, gx, gy, gx, gy, reward)

    # reset map
        if dataidx % 10 == 0:
            print("reseting map \n")
            #reset_gazebo()
            #time.sleep(1.0)
            reset_cmd = 'rosservice call /octomap_server/reset "{}" '
            os.system(reset_cmd)
            #reset_octomap()
            time.sleep(0.5)
            #listener.clear()
    # end loop condition
        if dataidx > MAX_NUM_DATA:
            print("max num data reached.. finishing the data generation \n")
            break

        # update data
        # previous_explored_area_size = explored_area_size
        # (wx_prev, wy_prev) = (wx_next, wy_next)
        # gmmsg_prev = gmmsg_next
        dataidx = dataidx + 1

# load comstmap


    # locate free cells (find possible next positions)

    #



    #rosrun gazebo_ros spawn_model -file ./turtlebot3_burger.urdf.xacro -urdf -model turtlebot3_burger -x -6 -y 6 -z 0


if __name__ == '__main__':
    main(sys.argv)
