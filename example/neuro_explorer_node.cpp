/*********************************************************************
Copyright 2024 The Ewha Womans University.
All Rights Reserved.
Permission to use, copy, modify OR distribute this software and its
documentation for educational, research and non-profit purposes, without
fee, and without a written agreement is hereby granted, provided that the
above copyright notice and the following three paragraphs appear in all
copies.
IN NO EVENT SHALL THE EWHA WOMANS UNIVERSITY BE
LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE EWHA WOMANS UNIVERSITY
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
THE EWHA WOMANS UNIVERSITY SPECIFICALLY DISCLAIM ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE EWHA WOMANS UNIVERSITY
HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
The authors may be contacted via:
Mail:        Y. J. Kim, Kyung Min Han
             Computer Graphics Lab
             Department of Computer Science and Engineering
             Ewha Womans University
             11-1 Daehyun-Dong Seodaemun-gu, Seoul, Korea 120-750
Phone:       +82-2-3277-6798
EMail:       kimy@ewha.ac.kr
             hankm@ewha.ac.kr
fee, and without a written agreement is hereby granted, provided that the
above copyright notice and the following three paragraphs appear in all
copies.

IN NO EVENT SHALL THE EWHA WOMANS UNIVERSITY BE
LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE EWHA WOMANS UNIVERSITY
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

THE EWHA WOMANS UNIVERSITY SPECIFICALLY DISCLAIM ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE EWHA WOMANS UNIVERSITY
HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


The authors may be contacted via:


Mail:        Y. J. Kim, Kyung Min Han
             Computer Graphics Lab                       
             Department of Computer Science and Engineering
             Ewha Womans University
             11-1 Daehyun-Dong Seodaemun-gu, Seoul, Korea 120-750


Phone:       +82-2-3277-6798


EMail:       kimy@ewha.ac.kr
             hankm@ewha.ac.kr
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include "neuro_explorer.hpp"

using namespace neuroexplorer;

enum SLAM_ID{GMAPPING=0, CARTOGRAPHER, SLAM_TOOLBOX, GT_MAPPING};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "neuro_explorer");
  const ros::NodeHandle nh;
  const ros::NodeHandle private_nh("~");

  ros::WallTime start_, end_;

  ROS_INFO("args: %s %s %s\n", argv[0], argv[1], argv[2]);

  string slam_method(argv[1]);
  int numthreads = atoi(argv[2]);
  static map<string, int> slamid;
  slamid["gmapping"] = SLAM_ID::GMAPPING;
  slamid["cartographer"] = SLAM_ID::CARTOGRAPHER;
  slamid["slam_toolbox"] = SLAM_ID::SLAM_TOOLBOX;
  slamid["gtmapping"]	= SLAM_ID::GT_MAPPING;

  //int totnumthreads = omp_get_max_threads() ;
  //ROS_INFO("max num threads, input num threads: %d %d \n", totnumthreads, numthreads);
  //numthreads = MIN( totnumthreads, numthreads );
  ROS_INFO("slamid: %s %d \n", slam_method.c_str(), slamid[slam_method.c_str()]);

  switch( slamid[slam_method.c_str()] )
  {
//  	  case GMAPPING:
//  	  {
//		  ROS_ERROR("Neuro-Explorer doesn't support GMAPPING yet. \n");
//		  return -1;
//  	  }

  	  case CARTOGRAPHER:
  	  {
		  ROS_ERROR("Neuro-Explorer is not compatible with CARTOGRAPHER yet. \n");
		  return -1;
  	  }
  	  case SLAM_TOOLBOX:
  	  {
		  ROS_INFO("Initializing neuro_explorer \n");
		  NeuroExplorer neuro_explorer(private_nh, nh);
		  neuro_explorer.SetNumThreads(numthreads);
		  ros::spinOnce();
		  //front_detector_dms.initmotion();
		  neuro_explorer.SetInitMotionCompleted();
		  while( !neuro_explorer.isDone() && ros::ok() )
		  {
			  try{
				  ros::spinOnce();
			  }
			  catch(std::runtime_error& e)
			  {
				ROS_ERROR("frontier_detector exception: %s", e.what());
				return -1;
			  }
		  }
		  //front_detector_dms.publishDone();
		  neuro_explorer.publishDoneExploration();
		  ros::Duration(0.5).sleep() ;
		  break;
  	  }

  	  case GT_MAPPING:
  	  {
		  ROS_INFO("Initializing neuro_explorer \n");
		  NeuroExplorer neuro_explorer(private_nh, nh);

		  ROS_INFO("Setting num of thread %d \n", numthreads);
		  neuro_explorer.SetNumThreads(numthreads);
		  ros::spinOnce();
		  neuro_explorer.initmotion( 0.2, 0.2, 1.0 );
		  neuro_explorer.SetInitMotionCompleted();
		  while( !neuro_explorer.isDone() && ros::ok() )
		  {
			  try{
				  ros::spinOnce();
			  }
			  catch(std::runtime_error& e)
			  {
				ROS_ERROR("frontier_detector exception: %s", e.what());
				return -1;
			  }
		  }
		  ROS_INFO("Finishing up this node \n");
		  //front_detector_dms.publishDone();
		  neuro_explorer.publishDoneExploration();
		  ros::Duration(0.5).sleep() ;
		  break;
  	  }

  	  default: ROS_ERROR("Invalid slam method \n");
  }

  ROS_WARN(" Shutting down the exploration task \n");
  //ros::shutdown(); lets shutdown from outside

  return 0;
}



