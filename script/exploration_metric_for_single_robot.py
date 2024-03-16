#! /usr/bin/env python

'''
subscribe /robot1/scan_map  /robot2/scan_map  /robot1/map  /robot2/map  /robot1/odom  /robot2/odom

'''
import sys
import time
import os
import numpy as np
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped
import tf
import pickle
import yaml
from PIL import Image
import math

exploration_rate_log = []
odom_log = []
path_length_log = []
time_log = []
end_flag = False
achieve_30 = False
achieve_40 = False
achieve_50 = False
achieve_60 = False
achieve_70 = False
achieve_80 = False
achieve_90 = False
achieve_95 = False
start_time = 0
curr_time = 0
last_msg_time = 0
gt_area = 0
begin_timing = False
num_robots = 1
T_report = np.ones([1, 9]) * np.inf # (30,40,50,60,70,80,90,95,99)
single_map_list = [OccupancyGrid() for i in range(num_robots)]
single_robot_coverage_rate_list = [0 for i in range(num_robots)]
res_dir = '/home/hankm/results/neuro_exploration_res'

def get_gt(pgm_file, yaml_file):
    map_img = np.array(Image.open(pgm_file))
    map_info = yaml.full_load(open(yaml_file, mode='r'))
    gt_area = (np.sum((map_img != 205).astype(int)))*map_info['resolution']*map_info['resolution']
    return gt_area

def callback(data):
    
    # -1:unkown 0:free 100:obstacle
    global end_flag, start_time, curr_time, achieve_30, achieve_40, achieve_50, achieve_60, achieve_70, achieve_80, achieve_90, achieve_95, last_msg_time, T_report
    
    msg_secs = data.header.stamp.secs
    now = rospy.get_time()
    
    #print("{} {}".format(msg_secs, now) )
    
    if (msg_secs + 3 < now):
        return
    else:
        last_msg_time = msg_secs

    
    if not end_flag:
        gridmap = np.array(data.data).reshape((data.info.height, data.info.width))
        explored_map = (gridmap != -1).astype(int)
        explored_area = explored_map.sum()*data.info.resolution*data.info.resolution
        exploration_rate = explored_area / gt_area
        exploration_rate_over_time = dict()
        exploration_rate_over_time['time'] = data.header.stamp
        exploration_rate_over_time['rate'] = exploration_rate
        curr_time = time.time()

        exploration_rate_log.append(exploration_rate_over_time)
        #print("exploration time: {} rate: {}".format( curr_time - start_time, exploration_rate ) )

        sys.stdout.write("exploration progress: %f%%   \r" % (exploration_rate * 100) )
        sys.stdout.flush()

        if exploration_rate >= 0.3 and (not achieve_30):
            t30 = curr_time - start_time
            print("achieve 0.3 coverage rate! \n")
            print("T_30: {}".format( t30 ) )
            achieve_30 = True
            T_report[:, 0] = t30
            
        if exploration_rate >= 0.4 and (not achieve_40):
            t40 = curr_time - start_time
            print("achieve 0.4 coverage rate! \n")
            print("T_40: {}".format( t40 ) )
            achieve_40 = True
            T_report[:, 1] = t40

        if exploration_rate >= 0.5 and (not achieve_50):
            t50 = curr_time - start_time
            print("achieve 0.5 coverage rate! \n")
            print("T_50: {}".format( t50 ) )
            achieve_50 = True
            T_report[:, 2] = t50

        if exploration_rate >= 0.6 and (not achieve_60):
            t60 = curr_time - start_time
            print("achieve 0.6 coverage rate!\n")
            print("T_60: {}".format( t60 ) )
            achieve_60 = True
            T_report[:, 3] = t60

        if exploration_rate >= 0.7 and (not achieve_70):
            t70 = curr_time - start_time
            print("achieve 0.7 coverage rate!\n")
            print("T_70: {}".format( t70 ) )
            achieve_70 = True
            T_report[:, 4] = t70

        if exploration_rate >= 0.8 and (not achieve_80):
            t80 = curr_time - start_time
            print("achieve 0.8 coverage rate!\n")
            print("T_80: {}".format( t80 ) )
            achieve_80 = True
            T_report[:, 5] = t80

        if exploration_rate >= 0.9 and (not achieve_90):
            t90 = curr_time - start_time
            print("achieve 0.9 coverage rate!\n")
            print("T_90: {}".format( t90) )
            achieve_90 = True
            T_report[:, 6] = t90

        if exploration_rate >= 0.95 and (not achieve_95):
            t95 = curr_time - start_time
            print("achieve 0.95 coverage rate!\n")
            print("T_95: {}".format(t95) )
            achieve_95 = True
            T_report[:, 7] = t95

        if exploration_rate >= 0.99:
            t99 = curr_time - start_time
            print("exploration ends!\n")
            print('T_total: %f  Cov_total %f' % (t99, exploration_rate) )
            T_report[:, 8] = t99
            print(T_report)
            outfile = '%s/coverage_time.txt' % res_dir
            print('saving time file to  {})\n'.format(outfile))
            np.savetxt(outfile, T_report, fmt='%6.2f')
            print('T report is written \n exploration_metric is finished \n')
            end_flag = True

            # # compute coverage std
            # coverage_std = np.std(np.array(single_robot_coverage_rate_list))
            # print("exploration coverage std: ", coverage_std)
            # # compute overlap rate
            # overlap_rate = np.sum(np.array(single_robot_coverage_rate_list)) - 1
            # print("exploration overlap rate: ", overlap_rate)
    else:
        # force to finish the processing
        print('end_flag is up .. finishing the recording  \n')
        pub = rospy.Publisher('exploration_is_done', Bool, queue_size=1)
        done_task = Bool()
        done_task.data = True
        pub.publish(done_task)
        #time.sleep(1)
    #else:
        #time.sleep(1)

def odom_callback(data):
    global end_flag
    if not end_flag:
        current_pos = [data.pose.pose.position.x, data.pose.pose.position.y]
        odom_over_time = dict()
        odom_over_time['time'] = data.header.stamp
        odom_over_time['odom'] = current_pos
        if len(odom_log) == 0:
            odom_log.append(odom_over_time)
            path_length_over_time = dict()
            path_length_over_time['time'] = data.header.stamp
            path_length_over_time['path_length'] = 0
            path_length_log.append(path_length_over_time)
        else:
            path_length_over_time = dict()
            path_length_over_time['time'] = data.header.stamp
            path_length_over_time['path_length'] = path_length_log[-1]['path_length'] + math.hypot(odom_log[-1]['odom'][0]-current_pos[0], odom_log[-1]['odom'][1]-current_pos[1])
            path_length_log.append(path_length_over_time)
            odom_log.append(odom_over_time)
        time.sleep(1)
    else:
        time.sleep(1)

def single_map_callback(data):
    global single_map_list
    # print(int(data.header.frame_id[5]))
    single_map_list[int(data.header.frame_id[5])-1] = data

def single_robot_coverage_rate_callback(data):
    global single_robot_coverage_rate_list, gt_area
    gridmap = np.array(data.data).reshape((data.info.height, data.info.width))
    explored_map = (gridmap != -1).astype(int)
    explored_area = explored_map.sum()*data.info.resolution*data.info.resolution
    exploration_rate = explored_area / gt_area
    single_robot_coverage_rate_list[int(data.header.frame_id[5])-1] = exploration_rate

def StartCallback(data):
    global start_time, begin_timing
    start_time = time.time()
    begin_timing = True
    print("Exploration start!")
    print("Start time: ", start_time)

def main(argv):
    global gt_area, T_report, curr_time, end_flag
    gt_area = get_gt(argv[1], argv[2]) 
    rospy.init_node('exploration_metric', anonymous=True)
    rospy.Subscriber("begin_exploration", Bool, StartCallback)
    rospy.Subscriber("map",  OccupancyGrid, callback, queue_size=1)
    #rospy.Subscriber("odom", Odometry, odom_callback, queue_size=1)
    #rospy.spin()
    while not rospy.is_shutdown():
        data = rospy.wait_for_message('exploration_is_done', Bool, timeout=None)
        #print("received exploration_is_done msg \n")
        if data.data is True or end_flag is True:
            break
    print("Finishing the exploration metric node \n")
    # print(T_report)
    # outfile = '%s/coverage_time.txt' % res_dir
    # np.savetxt(outfile, T_report, fmt='%6.2f')
    # print('T report is written \n exploration_metric is finished \n')

if __name__ == '__main__':
    main(sys.argv)

