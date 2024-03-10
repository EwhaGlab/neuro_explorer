#! /usr/bin/env python
import nav_msgs.msg
import rospy
import sys
import os
import random
import time
import numpy as np
import tf2_ros
import tf
from std_srvs.srv import Empty
from std_msgs.msg import Bool
import pdb
from scipy.io import savemat
import cv2
import roslaunch
from roslaunch.parent import ROSLaunchParent

def main(argv):
    workdir = os.getcwd()
    resdir = '/home/hankm/results/neuro_exploration_res/exploration_runs'

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch_world = 'willowgarage_x2.launch' #['room.launch', 'corner.launch', 'corridor.launch', 'loop_with_corridor.launch']
    worlds = ['room_with_corner', 'loop'] #['room', 'corner', 'corridor', 'loop_with_corridor']

    #widx = 1
    rospy.init_node('neuro_explorer_launcher', anonymous=True)

    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)

    start = time.time()
    weight = 0.8 # you need to put the best lambda here.
    numrounds = 5

    for ridx in range(0, numrounds):
        # make data dir
        print("processing %d th round with lambda = %f\n" % (ridx, weight) )
        savedir = resdir
#        cmd = 'mkdir -p %s' % savedir
#        os.system(cmd)

        # launch files
        # cli_args1 = ['autoexplorer', launch_worlds[widx]]
        # cli_args2 = ['autoexplorer', 'explore_bench.launch']
        # cli_args3 = ['autoexplorer', 'autoexplorer.launch']
        # roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(cli_args1)[0]
        # roslaunch_file2 = roslaunch.rlutil.resolve_launch_arguments(cli_args2)[0]
        # roslaunch_file3 = roslaunch.rlutil.resolve_launch_arguments(cli_args3)[0]
        #
        # launch_files = [roslaunch_file1, roslaunch_file2, roslaunch_file3]
        # # launch all files
        #
        # # launch move_base
        # parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        # parent.start()
        roslaunch.configure_logging(uuid)
        #launch0 = roslaunch.parent.ROSLaunchParent(uuid, [], is_core=True)
        #launch0.start()
        #time.sleep(2)
        launch1 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/hankm/catkin_ws/src/neuro_explorer/launch/includes/%s"%launch_world])
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/hankm/catkin_ws/src/neuro_explorer/launch/includes/explore_bench.launch"])

        weight_string = 'lambda:=%f' % weight
        cli_args = ["/home/hankm/catkin_ws/src/neuro_explorer/launch/neuro_explorer.launch", weight_string]
        launch3_args = cli_args[1:]
        launch3_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], launch3_args)]
        launch3 = roslaunch.parent.ROSLaunchParent(uuid, launch3_file)

        launch1.start()
        rospy.loginfo("world started")
        time.sleep(5)
        t = rospy.Time(0)
        (trans, rot) = listener.lookupTransform("odom", "base_link", t)
        print('launch cmd 1 is done \n')

        launch2.start()
        time.sleep(1)
        rospy.wait_for_message('map', nav_msgs.msg.OccupancyGrid, timeout=None)
        print('launch cmd 2 is done \n')

        launch3.start()
        print('launch cmd 3 is done \n')

        while not rospy.is_shutdown():
            data = rospy.wait_for_message('exploration_is_done', Bool, timeout=None)
            #print("exploration done? %d" % data.data)
            if data.data is True:
                break
        launch3.shutdown()
        print("Launch3 is shut down\n")
        launch2.shutdown()
        print("Launch2 is shut down\n")
        launch1.shutdown()
        print("Launch1 is shut down\n")
        #launch0.shutdown()
        #print('roscore is shutdown \n')
        time.sleep(60)

        # mv all res to the datadir
        cmd = 'mv %s/coverage_time.txt %s/ne_round%d.txt' % (resdir, savedir, ridx)
        os.system(cmd)

    end = time.time()
    print("tot exploration time ",  (end - start) )

if __name__ == '__main__':
    main(sys.argv)
