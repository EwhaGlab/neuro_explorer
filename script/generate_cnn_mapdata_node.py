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


def main(argv):
    workdir = os.getcwd()
    resdir = '/home/hankm/results/neuro_ffp'

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch_worlds = ['room.launch', 'corner.launch', 'corridor.launch', 'loop_with_corridor.launch']
    worlds = ['room_with_corner', 'loop'] #['room', 'corner', 'corridor', 'loop_with_corridor']

    #widx = 1
    rospy.init_node('autoexplorer_launcher', anonymous=True)

    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)

    start = time.time()


    for ii in range(0, 5):
        for widx in range(0, len(worlds)):
            # make data dir
            print("processing %d th data generation\n" % ii)
            savedir = '%s/%s/round%04d' % (resdir, worlds[widx], ii)
            cmd = 'mkdir -p %s' % savedir
            os.system(cmd)

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
            launch1 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/hankm/catkin_ws/src/autoexplorer/launch/includes/%s"%launch_worlds[widx]])
            launch2 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/hankm/catkin_ws/src/autoexplorer/launch/includes/explore_bench.launch"])
            launch3 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/hankm/catkin_ws/src/autoexplorer/launch/autoexplorer.launch"])

            launch1.start()
            rospy.loginfo("world started")
            time.sleep(5)
            t = rospy.Time(0)
            (trans, rot) = listener.lookupTransform("odom", "base_link", t)

            launch2.start()
            time.sleep(1)
            rospy.wait_for_message('map', nav_msgs.msg.OccupancyGrid, timeout=None)

            launch3.start()

            while not rospy.is_shutdown():
                data = rospy.wait_for_message('exploration_is_done', Bool, timeout=None)
                print("exploration done? %d" % data.data)
                if data.data is True:
                    break

            launch3.shutdown()
            launch2.shutdown()
            launch1.shutdown()
            #time.sleep(5)

            # mv all res to the datadir
            cmd = 'mv %s/*.* %s/'% (resdir, savedir)
            os.system(cmd)

    print("hello")
    end = time.time()
    print("data gen time ",  (end - start) )

if __name__ == '__main__':
    main(sys.argv)
