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


#ifndef INCLUDE_NEURO_EXPLORER_HPP_
#define INCLUDE_NEURO_EXPLORER_HPP_

#include "explorer.hpp"
#include "frontier_point.hpp"
#include "frontier_filter.hpp"
//#include "global_planning_handler.hpp"
#include <omp.h>
#include <algorithm>
#include "std_msgs/Empty.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <boost/format.hpp>
#include <fstream>

#include "image_data_handler.hpp"
#include "tensorflow/c/c_api.h"
#include "geometry_msgs/Point32.h"
#include "std_msgs/Bool.h"
#include "neuro_explorer/VizDataStamped.h"

#define DIST_HIGH  (1.0e10)
//#define DEBUG_MODE
//#define DATA_COLLECTION_MODE // data collection (gmap, rpos, frontier points) // this dataset is useful for timing experiment btwn FR-Net, A*-Net vs. DFFP, single-thread-A* planner
//#define SAVE_DNN_OUTPUTS

namespace neuroexplorer
{

bool class_cmp(const PointClass& a, const PointClass& b )
{
	if (a.label >= b.label) return true; // label based decending order
	else
		return false;
}

enum PrevExpState {
  ABORTED = 0,
  SUCCEEDED
};
static const char *prevstate_str[] =
        { "ABORTED", "SUCCEEDED" };



class NeuroExplorer: public Explorer
{
public:
	NeuroExplorer(const ros::NodeHandle private_nh_, const ros::NodeHandle &nh_);
	virtual ~NeuroExplorer();

	void initalize_fd_model( ) ;
	void initalize_astar_model( ) ;
	void initalize_covrew_model( ) ;

	void initmotion( const float& fvx, const float& fvy, const float& ftheta );
	inline void SetInitMotionCompleted(){ mb_isinitmotion_completed = true;  }
	inline void SetNumThreads(int numthreads){ mn_numthreads = numthreads; }

	void initGlobalmapimgs(  const int& cmheight, const int& cmwidth, const nav_msgs::OccupancyGrid& globalcostmap  );
	void copyFRtoGlobalmapimg(  const cv::Rect& roi_active_ds, const cv::Mat& fr_img );
	int  locateGlobalFRnFptsFromGlobalFRimg( const cv::Mat& FR_in_glob, const cv::Point& rpos_gmds, const cv::Point& am_orig_ds, vector<vector<cv::Point>>& contours_gm, vector<FrontierPoint>& fpts_gm ) ;
	int  locateFptsFromPredimg( const cv::Mat& potmap_prediction, const cv::Mat& covrew_predection, const vector<cv::Point>& FRcents_gm, const cv::Point& rpos_gmds, const cv::Point& amds_roi_orig,  vector<FrontierPoint>& fpts_gm ) ;
	int locateFRfromFRimg( const cv::Mat& cvFRimg, const cv::Point& rpos_gmds, const cv::Point& am_orig_ds, vector<vector<cv::Point>>& contours_gm, vector<cv::Point>& FRcents_gm );
	void globalCostmapCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg ) ;
	void gridmapCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg ) ;
	void globalCostmapUpdateCallback(const map_msgs::OccupancyGridUpdate::ConstPtr& msg );
	void robotPoseCallBack( const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg ) ;
	void robotVelCallBack( const geometry_msgs::Twist::ConstPtr& msg);
	void doneCB( const actionlib::SimpleClientGoalState& state ) ;

	void mapdataCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg); //const octomap_server::mapframedata& msg ) ;
	void gobalPlanCallback(const visualization_msgs::Marker::ConstPtr& msg) ;
	//virtual void moveRobotCallback(const nav_msgs::Path::ConstPtr& msg ) ;
	void moveRobotCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg ) ;
	void unreachablefrontierCallback(const geometry_msgs::PoseStamped::ConstPtr& msg );

	void setVizMarkerFromPointClass( const PointClassSet& pointset, visualization_msgs::Marker& vizmarker, const rgb& init_color, float fsize );
	void publishDoneExploration() ;
	void publishVizMarkers( bool bviz_flag = true );
	void appendUnreachablePoint( const geometry_msgs::PoseStamped& unreachablepose ) ;

	void updatePrevFrontierPointsList( );
//	geometry_msgs::PoseStamped StampedPosefromSE2( float x, float y, float yaw ) ;
//	geometry_msgs::PoseStamped GetCurrPose ( ) ;

	cv::Point world2gridmap( cv::Point2f img_pt_roi );
	cv::Point2f gridmap2world( cv::Point grid_pt );
	void generateGridmapFromCostmap( ) ;

	int saveMap( const nav_msgs::OccupancyGrid& map, const string& infofilename, const string& mapfilename ) ;
	int saveFrontierPoints( const nav_msgs::OccupancyGrid& map, const nav_msgs::Path& msg_frontiers, int bestidx, const string& frontierfile  ) ;
	int savefrontiercands( const nav_msgs::OccupancyGrid& map, const vector<FrontierPoint>& voFrontierPoints, const string& frontierfile ) ;

	int frontier_summary( const vector<FrontierPoint>& voFrontierCurrFrame );

	void updateUnreachablePointSet(  const nav_msgs::OccupancyGrid& globalcostmap  ) ;

	int sort_by_distance_to_robot( const geometry_msgs::PoseStamped& robotpose, vector<geometry_msgs::PoseStamped>& frontierpoints );
	int selectNextClosestPoint( const geometry_msgs::PoseStamped& robotpose, const vector<geometry_msgs::PoseStamped>& vmsg_frontierpoints, geometry_msgs::PoseStamped& nextbestpoint  ) ;
	int selectNextBestPoint( const vector<geometry_msgs::PoseStamped>& vmsg_frontierpoints, int nbestidx, geometry_msgs::PoseStamped& nextbestpoint  ) ;

	int selectEscapingPoint( geometry_msgs::PoseStamped& escapepoint) ;
	int moveBackWard() ;

	// save DNN data sets for DNN
	void saveDNNData( const cv::Mat& img_frontiers_offset, const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& best_goal,
						const std::vector<geometry_msgs::PoseStamped>& best_plan, const int& OFFSET, const cv::Rect& roi );

	cv::Point compute_rpose_wrt_maporig( )
	{
		float frx_w = m_init_robot_pose.pose.position.x ;
		float fry_w = m_init_robot_pose.pose.position.y ;
		return world2gridmap( cv::Point2f( frx_w, fry_w ) ) ;
	}

	geometry_msgs::PoseStamped GetCurrRobotPose ( )
	{
		tf::StampedTransform map2baselink;
		try{
		  m_listener.lookupTransform(m_worldFrameId, m_baseFrameId,
								   ros::Time(0), map2baselink);
		}
		catch (tf::TransformException &ex) {
		  ROS_ERROR("%s",ex.what());
		  ros::Duration(1.0).sleep();
		}

		geometry_msgs::PoseStamped outPose;
		outPose.pose.position.x = map2baselink.getOrigin().x();
		outPose.pose.position.y = map2baselink.getOrigin().y();
		outPose.pose.position.z = 0.f;
		outPose.header.frame_id = m_worldFrameId;

		return outPose;
	}

	inline bool frontier_sanity_check( int nx, int ny, int nwidth, const std::vector<signed char>& cmdata )
	{
		// 0	1	2
		// 3	4	5
		// 6	7	8

		int i0 = nwidth * (ny - 1) 	+	nx - 1 ;
		int i1 = nwidth * (ny - 1) 	+	nx 		;
		int i2 = nwidth * (ny - 1) 	+	nx + 1	;
		int i3 = nwidth * ny			+	nx - 1 ;
		int i5 = nwidth * ny			+	nx + 1 ;
		int i4 = nwidth * ny			+ 	nx ;
		int i6 = nwidth * (ny + 1)	+	nx - 1 ;
		int i7 = nwidth * (ny + 1)	+	nx		;
		int i8 = nwidth * (ny + 1)	+	nx + 1 ;
		int nleathalcost = 70 ;

		//ROS_INFO("width i0 val : %d %d %d\n", nwidth, i0, gmdata[i0] );
		if( cmdata[i0] > nleathalcost || cmdata[i1] > nleathalcost || cmdata[i2] > nleathalcost || cmdata[i3] > nleathalcost ||
			cmdata[i5] > nleathalcost || cmdata[i6] > nleathalcost || cmdata[i7] > nleathalcost || cmdata[i8] > nleathalcost )
		{
			return false;
		}

		if( cmdata[i4] < 0 )
		{
			return true ;
		}
		else
		{
			return false ;
		}
	}

	inline bool is_explored( int nx, int ny, int nwidth, const std::vector<signed char>& gmdata )
	{
		// 0	1	2
		// 3		5
		// 6	7	8

		int i0 = nwidth * (ny - 1) 	+	nx - 1 ;
		int i1 = nwidth * (ny - 1) 	+	nx 		;
		int i2 = nwidth * (ny - 1) 	+	nx + 1	;
		int i3 = nwidth * ny			+	nx - 1 ;
		int i5 = nwidth * ny			+	nx + 1 ;
		int i6 = nwidth * (ny + 1)	+	nx - 1 ;
		int i7 = nwidth * (ny + 1)	+	nx		;
		int i8 = nwidth * (ny + 1)	+	nx + 1 ;

		if( gmdata[i0] < 0 || gmdata[i1] < 0 || gmdata[i2] < 0 || gmdata[i3] < 0 ||  gmdata[i5] < 0 || gmdata[i6] < 0 || gmdata[i7] < 0 || gmdata[i8] < 0 )
		{
			return false ;
		}
		else
		{
			return true ;
		}
	}

	inline bool is_frontier_point( int nx, int ny, int nwidth, int nheight, const std::vector<signed char>& gmdata )
	{
		// 0	1	2
		// 3		5
		// 6	7	8

		if( nx < 1 || nx > nwidth-2 || ny < 1 || ny > nheight -2 )  // this point is at the border of the map
			return false;

		int i0 = nwidth * (ny - 1) 	+	nx - 1 ;
		int i1 = nwidth * (ny - 1) 	+	nx 		;
		int i2 = nwidth * (ny - 1) 	+	nx + 1	;
		int i3 = nwidth * ny			+	nx - 1 ;
		int i5 = nwidth * ny			+	nx + 1 ;
		int i6 = nwidth * (ny + 1)	+	nx - 1 ;
		int i7 = nwidth * (ny + 1)	+	nx		;
		int i8 = nwidth * (ny + 1)	+	nx + 1 ;

		if( gmdata[ nwidth*ny + nx ] != -1 )
		{
			// nx and ny must be UNKNOWN
			false;
		}
		else if( gmdata[i0] < 0 && gmdata[i1] < 0 && gmdata[i2] < 0 && gmdata[i3] < 0 &&  gmdata[i5] < 0 && gmdata[i6] < 0 && gmdata[i7] < 0 && gmdata[i8] < 0 )
		{
			// at least one of its neighboring point must be FREE
			return false ;
		}
		else
		{
			return true ;
		}
	}


    inline bool equals_to_prevgoal( const geometry_msgs::PoseStamped& in_goal )
    {
//    	  ROS_WARN("prev/curr goal (%f %f), (%f %f) \n",
//    			  previous_goal_.pose.position.x, previous_goal_.pose.position.y,
//				  planner_goal_.pose.position.x, planner_goal_.pose.position.y
//    	  );

  	  float fxdiff = (m_previous_goal.pose.pose.position.x - in_goal.pose.position.x) ;
  	  float fydiff = (m_previous_goal.pose.pose.position.y - in_goal.pose.position.y) ;
  	  return std::sqrt( fxdiff * fxdiff + fydiff * fydiff ) < 0.001 ? true : false;
    }

    inline void world_to_scaled_gridmap( float fwx, float fwy, float fox, float foy, float fres, int nscale, int& nmx, int& nmy  )
    {
    	int ngmx = static_cast<int>( (fwx - fox) / fres ) ;
    	int ngmy = static_cast<int>( (fwy - foy) / fres ) ;
    	nmx = ngmx / nscale ;
    	nmy = ngmy / nscale ;
    }

    inline cv::Point gridmapDS_to_activemap( const cv::Point& pt_gmds, const cv::Point& rpos_gmds )
    {
    	int nx_am = ( pt_gmds.x - rpos_gmds.x ) + mn_cnn_width / 2;
    	int ny_am = ( pt_gmds.y - rpos_gmds.y ) + mn_cnn_height/ 2;
    	return cv::Point(nx_am, ny_am);
    }

    inline cv::Point activemap_to_gridmapDS( const cv::Point& pt_am, const cv::Point& rpos_gmds )
    {
    	int nx_gmds = ( pt_am.x - mn_cnn_width / 2) + rpos_gmds.x ;
    	int ny_gmds = ( pt_am.y - mn_cnn_height/ 2) + rpos_gmds.y ;
    	return cv::Point(nx_gmds, ny_gmds) ;
    }

    inline cv::Point gridmapDS_to_globalmapDS( const cv::Point& pt_gmds, const cv::Point& rpos_gmds, const cv::Point& am_roi_orig )
    {
    	cv::Point pt_am = gridmapDS_to_activemap( pt_gmds, rpos_gmds );
    	return cv::Point( pt_am.x + am_roi_orig.x, pt_am.y + am_roi_orig.y ) ;
    }

    cv::Point globalmapDS_to_gridmapDS( const cv::Point& pt_glob_ds, const cv::Point& rpos_gmds, const cv::Point& am_roi_orig )
    {
    	cv::Point pt_am = cv::Point( pt_glob_ds.x - am_roi_orig.x, pt_glob_ds.y - am_roi_orig.y );
    	cv::Point pt_gmds = activemap_to_gridmapDS( pt_am, rpos_gmds );
    	return pt_gmds ;
    }

    static float euc_dist(const cv::Point2f& lhs, const cv::Point2f& rhs)
    {
        return (float)sqrt(  (lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y)  );
    }

    static float euc_dist(const geometry_msgs::PoseStamped& lhs, const geometry_msgs::PoseStamped& rhs)
    {
        return (float)sqrt(  (lhs.pose.position.x - rhs.pose.position.x) * (lhs.pose.position.x - rhs.pose.position.x) + (lhs.pose.position.y - rhs.pose.position.y) * (lhs.pose.position.y - rhs.pose.position.y)  );
    }

// tensorflow apis
    static void NoOpDeallocator(void* data, size_t a, void* b){};
    void run_tf_fr_detector_session( const cv::Mat& input_map, cv::Mat& model_output ) ;
    void run_tf_astar_session( const cv::Mat& input_map, cv::Mat& model_output );
    void run_tf_covrew_session( const cv::Mat& input_map, cv::Mat& model_output );

// optimal FR pt selection
    int locate_optimal_point_from_potmap( const cv::Mat& input_potmap, const uint8_t& optVal, vector<cv::Point>& points   ) ;
    int assign_classes_to_points( const cv::Mat& input_map, vector<PointClass>& points   ) ;

//	Ensemble
    void ensemble_predictions( const cv::Mat& potmap_prediction, const cv::Mat& covrew_prediction, cv::Mat& ensembled_output );

protected:

	ros::NodeHandle m_nh;
	ros::NodeHandle m_nh_private;

	ros::Subscriber 	m_mapSub, m_poseSub, m_velSub, m_mapframedataSub, m_globalCostmapSub, m_globalCostmapUpdateSub, m_frontierCandSub,
						m_currGoalSub, m_globalplanSub, m_unreachablefrontierSub ;
	ros::Publisher 		m_currentgoalPub, m_targetsPub, m_unreachpointPub, m_velPub, m_donePub, m_resetgazeboPub, m_startmsgPub,
						m_otherfrontierptsPub, m_vizDataPub ;

	int mn_numthreads;
	int mn_globalcostmapidx ;
	string mstr_inputparams ;
	bool mb_isinitmotion_completed ;
	int mn_cnn_height, mn_cnn_width ;  	// down sampled img for DNN network
	cv::Mat mcvu_globalmapimg, mcvu_costmapimg, mcvu_mapimgroi ;
	cv::Mat mcvu_globalmapimg_ds, mcvu_globalfrimg_ds ;		// down sampled global FR img

	FrontierFilter mo_frontierfilter;
	tf::TransformListener m_listener;

	//GlobalPlanningHandler* mpo_gph ;
	//GlobalPlanningHandler mo_gph ;
	costmap_2d::Costmap2D* mpo_costmap;
	bool mb_allow_unknown ;

	uint8_t* mp_cost_translation_table;

	ofstream m_ofs_time ;
	float mf_neighoringpt_decisionbound ;
	bool mb_strict_unreachable_decision ;

	geometry_msgs::PoseWithCovarianceStamped m_previous_goal ;
	PrevExpState me_prev_exploration_state ;
	//int mn_prev_nbv_posidx ;
	bool mb_nbv_selected ;
	ros::Time m_last_oscillation_reset ;
	geometry_msgs::PoseStamped m_previous_robot_pose ;
	geometry_msgs::PoseStamped m_init_robot_pose ;

	vector<FrontierPoint> mvo_globalfpts_gm, 	mvo_localfpts_gm ;
	vector<vector<cv::Point>> mvvo_globalfr_gm,	mvvo_localfr_gm ;

// tensorflow api
// frontier detection
	//TF_Graph* mptf_fd_Graph;
	TF_Status* mptf_fd_Status ;
    TF_SessionOptions* mptf_fd_SessionOpts ;
    TF_Buffer* mptf_fd_RunOpts ;
    TF_Session* mptf_fd_Session;
    string m_str_fd_modelfilepath;
    TF_Output* mptf_fd_input ;
    TF_Output mtf_fd_t0 ;
    TF_Output* mptf_fd_output ;
    TF_Output mtf_fd_t2 ;
    TF_Tensor** mpptf_fd_input_values ;
    TF_Tensor** mpptf_fd_output_values ;
    TF_Tensor* mptf_fd_int_tensor ;
    float* mpf_fd_data ;

// astar potmap predictor network
	TF_Graph* mptf_astar_Graph;
	TF_Status* mptf_astar_Status ;
    TF_SessionOptions* mptf_astar_SessionOpts ;
    TF_Buffer* mptf_astar_RunOpts ;
    TF_Session* mptf_astar_Session;
    string m_str_astar_modelfilepath;
    TF_Output* mptf_astar_input ;
    TF_Output mtf_astar_t0 ;
    TF_Output* mptf_astar_output ;
    TF_Output mtf_astar_t2 ;
    TF_Tensor** mpptf_astar_input_values ;
    TF_Tensor** mpptf_astar_output_values ;
    TF_Tensor* mptf_astar_int_tensor ;
    float* mpf_astar_data ;

// covrew predictor network
	//TF_Graph* mptf_covrew_Graph;
	TF_Status* mptf_covrew_Status ;
	TF_SessionOptions* mptf_covrew_SessionOpts ;
	TF_Buffer* mptf_covrew_RunOpts ;
	TF_Session* mptf_covrew_Session;
	string m_str_covrew_modelfilepath;
	TF_Output* mptf_covrew_input ;
	TF_Output mtf_covrew_t0 ;
	TF_Output* mptf_covrew_output ;
	TF_Output mtf_covrew_t2 ;
	TF_Tensor** mpptf_covrew_input_values ;
	TF_Tensor** mpptf_covrew_output_values ;
	TF_Tensor* mptf_covrew_int_tensor ;
	float* mpf_covrew_data ;

    int mn_num_classes ;
    geometry_msgs::PoseStamped m_rpos_world ;

// gridmap, activmap, globalmap tforms....
    cv::Point m_rpos_gm ;
    cv::Rect m_roi_active_ds ;

private:
	std::mutex mutex_robot_state;
	std::mutex mutex_unreachable_frontier_set;

	std::mutex mutex_gridmap;
	std::mutex mutex_costmap;
	std::mutex mutex_upperbound;
	std::mutex mutex_timing_profile;
	std::mutex mutex_currgoal ;

	omp_lock_t m_mplock;
	ImageDataHandler m_data_handler;

	// weighted sum coefficients used for ensembling.
	float mf_lambda ;

	// for debug
	ofstream m_ofs_logger ;
	ofstream m_ofs_ae_report ;
	string mstr_report_filename ;

	int mn_mapcallcnt, mn_mapdatacnt, mn_moverobotcnt ;
	double mf_avgcallbacktime_msec, mf_totalcallbacktime_msec ;

	//double mf_avgplanngtime_msec, mf_totalplanningtime_msec  ;
	//double mf_avgffptime_msec, mf_totalffptime_msec ;
	//double mf_avgcovrewtime_msec, mf_totalcovrewtime_msec ;

	double mf_avgmotiontime_msec, mf_totalmotiontime_msec ;

	double mf_avg_fd_sessiontime_msec, mf_total_fd_sessiontime_msec;
	double mf_avg_astar_sessiontime_msec, mf_total_astar_sessiontime_msec;
	double mf_avg_covrew_sessiontime_msec, mf_total_covrew_sessiontime_msec;

	vector<double> mvf_fr_detection_time, mvf_astar_time, mvf_covrew_time;
	bool mn_zero_FR_incident_cnts ;

	ros::WallTime m_ae_start_time ;
};

}





#endif /* INCLUDE_NEURO_EXPLORER_HPP_ */
