/*********************************************************************
Copyright 2024 The Ewha Womans University.
All Rights Reserved.
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
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
*/

#ifndef INCLUDE_EXPLORER_HPP_
#define INCLUDE_EXPLORER_HPP_


#define OCCUPIED_BIN_THR 	(128)
#define WEAK_COMPONENT_THR	(10)
#define MIN_TARGET_DIST		(30) // to prevent relocating the current pose as the frontier point over and over again
//#define LEATHAL_COST_THR	(80)
//#define FD_DEBUG_MODE

#include <ros/console.h>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "octomap_server/mapframedata.h"

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>

// map saver
#include "map_server/image_loader.h"
#include "nav_msgs/MapMetaData.h"
#include "yaml-cpp/yaml.h"

#include "nav_msgs/OccupancyGrid.h"
#include "map_msgs/OccupancyGridUpdate.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "std_msgs/Header.h"
#include "std_msgs/Bool.h"
#include "nav_msgs/MapMetaData.h"
#include "geometry_msgs/Point.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include <tf/transform_listener.h>
#include <fstream>

#include <move_base/move_base.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/action_client.h>

#include <tf/transform_listener.h>
#include "ros/service_client.h"
#include "nav_msgs/GetPlan.h"

#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <experimental/filesystem>
#include <set>
#include "frontier_point.hpp"

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> SimpleMoveBaseClient ;

typedef struct _FrontierInfo {
	cv::Point location ;
	int revenue ;
}FrontierInfo ;

namespace neuroexplorer
{


typedef enum{	ROBOT_IS_NOT_MOVING 	= -1,
				ROBOT_IS_READY_TO_MOVE	= 0,
				FORCE_TO_STOP    		= 1, // is moving but needs to be stopped
				ROBOT_IS_MOVING  		= 2
			} ROBOT_STATE ;

static const char *robot_state[] =
	  { "ROBOT_IS_NOT_MOVING", "ROBOT_IS_READY_TO_MOVE", "FORCE_TO_STOP", "ROBOT_IS_MOVING" };

class Explorer
{
public:
	//FrontierDetector(const ros::NodeHandle private_nh_, const ros::NodeHandle &nh_);
	Explorer() ;
	virtual ~Explorer();

	virtual void mapdataCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg); //const octomap_server::mapframedata& msg ) ;
	virtual vector<cv::Point> eliminateSupriousFrontiers( nav_msgs::OccupancyGrid &costmapData, vector<cv::Point> frontierCandidates, int winsize );

	bool correctFrontierPosition( const nav_msgs::OccupancyGrid &gridmap, const cv::Point& frontierCandidate, const int& winsize, cv::Point& correctedPoint );
	//virtual void accessFrontierPoint( ) ;

	virtual cv::Point2f img2gridmap( cv::Point img_pt );
	virtual cv::Point gridmap2img( cv::Point2f grid_pt );

//	virtual void globalCostmapCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg);
//	virtual void robotPoseCallBack( const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg );
//	virtual void robotVelCallBack( const geometry_msgs::Twist::ConstPtr& msg );

	const int nccxidx[6] = {0,-1,0,1} ;
	const int nccyidx[6] = {-1,0,1,0} ;

	float Norm(cv::Point2f x1, cv::Point2f x2)
	{
		return pow(	(pow((x2.x-x1.x),2)+pow((x2.y-x1.y),2))	,0.5);
	}

	void downSampleMap( cv::Mat& uImage )
	{
		// Labeling is a bit weird but works OK with this way.
		// unknown 255, occupied 127, free 0

		cv::Mat uOccu = uImage.clone();
		cv::threshold( uOccu, uOccu, 187, 255, cv::THRESH_TOZERO_INV ); 	// 187 ~ 255 --> 0
		cv::threshold( uOccu, uOccu, 67,  255, cv::THRESH_TOZERO ); 		// 0 ~ 66 	--> 0
		cv::threshold( uOccu, uOccu, 0, 255, cv::THRESH_BINARY) ;// 67 ~ 187  --> 255 (occupied)

		cv::Mat uUnkn = uImage.clone();
		cv::threshold( uUnkn, uUnkn, 187, dffp::MapStatus::UNKNOWN, cv::THRESH_BINARY ); // 187 ~ 255 --> 255

		for(int iter=0; iter < mn_numpyrdownsample; iter++ )
		{
			pyrDown(uOccu, uOccu, cv::Size(uOccu.rows/2, uOccu.cols/2));
			pyrDown(uUnkn, uUnkn, cv::Size(uUnkn.rows/2, uUnkn.cols/2));
		}

		cv::threshold(uOccu, uOccu, 0, dffp::MapStatus::OCCUPIED, CV_THRESH_BINARY) ;
		cv::threshold(uUnkn, uUnkn, 0, dffp::MapStatus::UNKNOWN, CV_THRESH_BINARY) ;
		uImage = uOccu + uUnkn ;
	}

	void clusterToThreeLabels( cv::Mat& uImage  )
	{
		cv::Mat uUnkn = uImage.clone();
		cv::threshold( uUnkn, uUnkn, 187, 255, cv::THRESH_TOZERO_INV ); 	// 187 ~ 255 --> 0
		cv::threshold( uUnkn, uUnkn, 67,  255, cv::THRESH_TOZERO ); 		// 0 ~ 66 	--> 0
		cv::threshold( uUnkn, uUnkn, 0, dffp::MapStatus::UNKNOWN, cv::THRESH_BINARY) ;// 67 ~ 187  --> 127 (unknown)

		cv::Mat uOcc = uImage.clone();
		cv::threshold( uOcc, uOcc, 128, dffp::MapStatus::OCCUPIED, cv::THRESH_BINARY ); // 187 ~ 255 --> 255
		uImage = uOcc + uUnkn ;
#ifdef SAVE_DEBUG_IMAGES
		cv::imwrite("/home/hankm/catkin_ws/src/frontier_detector/launch/uImage.png",  uImage);
		cv::imwrite("/home/hankm/catkin_ws/src/frontier_detector/launch/occ.png",  uOcc);
		cv::imwrite("/home/hankm/catkin_ws/src/frontier_detector/launch/unkn.png", uUnkn);
#endif
	}

	geometry_msgs::PoseStamped StampedPosefromSE2( const float& x, const float& y, const float& yaw_radian )
	{
		geometry_msgs::PoseStamped outPose ;
		outPose.pose.position.x = x ;
		outPose.pose.position.y = y ;

		float c[3] = {0,};
		float s[3] = {0,};
		c[0] = cos(yaw_radian/2) ;
		c[1] = cos(0) ;
		c[2] = cos(0) ;
		s[0] = sin(yaw_radian/2) ;
		s[1] = sin(0) ;
		s[2] = sin(0) ;

		float qout[4] = {0,};
		qout[0] = c[0]*c[1]*c[2] + s[0]*s[1]*s[2];
		qout[1] = c[0]*c[1]*s[2] - s[0]*s[1]*c[2];
		qout[2] = c[0]*s[1]*c[2] + s[0]*c[1]*s[2];
		qout[3] = s[0]*c[1]*c[2] - c[0]*s[1]*s[2];

		outPose.pose.orientation.w = qout[0] ;
		outPose.pose.orientation.x = qout[1] ;
		outPose.pose.orientation.y = qout[2] ;
		outPose.pose.orientation.z = qout[3] ;

		outPose.header.frame_id = m_worldFrameId ;
		outPose.header.stamp = ros::Time::now() ;

		return outPose;
	}

	geometry_msgs::PoseStamped GetCurrPose ( )
	{
		geometry_msgs::PoseStamped outPose ;
		outPose.header = m_robotpose.header ;
		outPose.pose.position.x = m_robotpose.pose.pose.position.x ;
		outPose.pose.position.y = m_robotpose.pose.pose.position.y ;
		outPose.pose.position.z = 0.f ;

		outPose.pose.orientation = m_robotpose.pose.pose.orientation ;

		return outPose;
	}

	inline bool isDone() const { return mb_explorationisdone; }

	void saveMetaData(const string& metadatafilename, const nav_msgs::MapMetaData& mapInfo, const geometry_msgs::PoseWithCovarianceStamped& rpos_w );
	void saveRobotPose(const string& rposefilename, const geometry_msgs::PoseWithCovarianceStamped& rpos_w );
	void saveGridmap( const string& mapfilename, const string& mapinfofilename, const nav_msgs::OccupancyGrid& mapData );
	void saveGridmap( const string& mapfilename, const nav_msgs::OccupancyGrid& mapData );
	void saveCostmap( const string& mapfilename, const nav_msgs::OccupancyGrid& mapData );

	void writeGridmapToPNG( const string& filename, const nav_msgs::OccupancyGrid &mapData );
	void saveFrontierCandidates( const string& filename, const vector<FrontierPoint>& voFrontierCandidates );

protected:

	string m_str_debugpath ;
	string m_str_inputparams ;
	bool mb_savemap ;
	cv::FileStorage m_fs;

	int mn_numpyrdownsample;
	int mn_scale;
	int mn_roi_size ;
	int mn_globalmap_width, mn_globalmap_height, mn_globalmap_centx, mn_globalmap_centy ; // global
	//int mn_activemap_width, mn_activemap_height ;
	float mf_resolution ;
	int mn_correctionwindow_width ;

	nav_msgs::OccupancyGrid m_gridmap;
	nav_msgs::OccupancyGrid m_globalcostmap ;
	nav_msgs::Path			m_pathplan ;

	geometry_msgs::PoseWithCovarianceStamped m_robotpose ; // (w.r.t world)
	geometry_msgs::Twist m_robotvel ;

	std::string m_worldFrameId;
	std::string m_mapFrameId;
	std::string m_baseFrameId ;

	int mn_cols, mn_rows, mn_globalmap_xc_, mn_globalmap_yc_ ;
	//int m_nCannotFindFrontierCount ;
	bool mb_explorationisdone ;

	vector<cv::Point> m_frontiers;
	int m_frontiers_region_thr ;
	uint32_t mu_cmheight, mu_cmwidth, mu_gmheight, mu_gmwidth ;

	actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> m_move_client ;
	ros::ServiceClient m_makeplan_client;
//	cv::Mat m_uMapImg, m_uMapImgROI ;

	geometry_msgs::PoseWithCovarianceStamped m_targetgoal, m_optimal_targetgoal ; // actual goal /  optimal goal
	set<pointset> m_unreachable_frontier_set ;

	// curr/prev (global, accumulated) frontier point set
	set<pointset> m_curr_acc_frontierset, m_prev_acc_frontierset ;

	// thrs
	//float	m_costmap_conf_thr ;
	//float	m_gridmap_conf_thr ;
	int	mn_occupancy_thr ; // 0 ~ 100
	int mn_lethal_cost_thr ;
	double mf_robot_radius ;

	ROBOT_STATE me_robotstate ;

};

}

#endif /* INCLUDE_EXPLORER_HPP_ */
