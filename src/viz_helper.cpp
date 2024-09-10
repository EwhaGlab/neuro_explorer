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


#include "viz_helper.hpp"

namespace neuroexplorer
{

VizHelper::VizHelper(const ros::NodeHandle private_nh_, const ros::NodeHandle &nh_):
m_nh_private(private_nh_),
m_nh(nh_),
mb_explorationisdone(false)
{
	m_nh.getParam("/neuroexplorer/debug_data_save_path", m_str_debugpath);

	m_nh.param("/neuroexplorer/global_width",  mn_globalmap_width, 	2048) ;
	m_nh.param("/neuroexplorer/global_height", mn_globalmap_height,	2048) ;

	m_nh.param("/neuroexplorer/num_downsamples", mn_numpyrdownsample, 0);
	m_nh.param("/neuroexplorer/frame_id", m_worldFrameId, std::string("map"));
	m_nh.param("/move_base/global_costmap/resolution", mf_resolution, 0.05f) ;
//	m_nh.param("move_base/global_costmap/robot_radius", mf_robot_radius, 0.12); // 0.3 for fetch

	m_nh.param("/tf_loader/cnn_width", mn_cnn_width, 512);
	m_nh.param("/tf_loader/cnn_height", mn_cnn_height, 512);

	mn_scale = pow(2, mn_numpyrdownsample);
	mn_vizbound_width  = mn_cnn_width  * mn_scale ;
	mn_vizbound_height = mn_cnn_height * mn_scale ;

	m_vizDataSub  	= m_nh.subscribe("viz_data", 1, &VizHelper::vizCallback, this); // kmHan
	m_expdoneSub	= m_nh.subscribe("exploration_is_done", 1, &VizHelper::doneCallback, this);

	//m_frontier_region_markers = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, 1.f, 0.f, 0.f, 1.f, 0.1 );
	//m_frontier_region_markers.type = visualization_msgs::Marker::POINTS;

	m_global_frontierpoint_markers = visualization_msgs::MarkerArray() ;

//	m_frontier_region =
//	, m_local_frontier_points, m_global_frontier_points, m_active_bound, m_target_goal;

	m_currentgoalPub = m_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("curr_goalpose", 10);
	m_makergoalPub = m_nh.advertise<visualization_msgs::Marker>("curr_goal_marker",10);
	m_markerfrontierPub = m_nh.advertise<visualization_msgs::MarkerArray>("frontier_point_markers", 10);
	m_markerglobalfrontierPub = m_nh.advertise<visualization_msgs::MarkerArray>("global_frontier_point_markers", 10);
	m_markerfrontierregionPub = m_nh.advertise<visualization_msgs::Marker>("frontier_region_markers", 1);
	m_marker_optcov_regionPub	  = m_nh.advertise<visualization_msgs::Marker>("cov_opt_markers",1);
	m_marker_optastar_regionPub	  = m_nh.advertise<visualization_msgs::Marker>("astar_opt_markers",1);
	m_marker_optensembled_regionPub = m_nh.advertise<visualization_msgs::Marker>("ensembled_opt_markers",1);
	m_active_boundPub				= m_nh.advertise<visualization_msgs::Marker>("active_bound_lines", 1);
	m_marker_unreachpointPub = m_nh.advertise<visualization_msgs::MarkerArray>("unreachable_marker", 10);

	mn_FrontierID = 1;
	mn_global_FrontierID = 1;
	mn_UnreachableFptID = 0;

}

VizHelper::~VizHelper()
{
}

void VizHelper::setActiveBound( const float& frx_w, const float& fry_w, const int& ngmwidth, const int& ngmheight, visualization_msgs::Marker& vizmarker )
{
	float fbound_size_world = static_cast<float>(ngmwidth) * m_vizdata.mapdata_info.resolution ;

	float fox_w = frx_w - fbound_size_world / 2 ;
	float foy_w = fry_w - fbound_size_world / 2 ;
	float fBx_w = fox_w + fbound_size_world ;
	float fBy_w = foy_w + fbound_size_world ;

	vizmarker = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, 1.f, 0.65, 0.f, 1.f, 0.2);
	vizmarker.type = visualization_msgs::Marker::LINE_STRIP;
	geometry_msgs::Point tl, tr, br, bl;
	tl.x = fox_w ;
	tl.y = foy_w ;
	tl.z = 0.f;

	tr.x = fBx_w ;
	tr.y = foy_w ;
	tr.z = 0.f;

	br.x = fBx_w;
	br.y = fBy_w;
	br.z = 0.f ;

	bl.x = fox_w ;
	bl.y = fBy_w ;
	bl.z = 0.f ;

	vizmarker.points.push_back(tl);
	vizmarker.points.push_back(tr);
	vizmarker.points.push_back(br);
	vizmarker.points.push_back(bl);
	vizmarker.points.push_back(tl);
}

void VizHelper::removeGlobalFrontierPointMarkers()
{
	visualization_msgs::MarkerArray global_ftmarkers_old = m_global_frontierpoint_markers ;
	for(size_t idx=0; idx < global_ftmarkers_old.markers.size(); idx++)
		global_ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
	m_markerglobalfrontierPub.publish(global_ftmarkers_old);
	m_global_frontierpoint_markers.markers.resize(0);
}

void VizHelper::removeLocalFrontierPointMarkers()
{
	visualization_msgs::MarkerArray ftmarkers_old = m_local_frontierpoint_markers ;
	for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
		ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
	m_markerfrontierPub.publish(ftmarkers_old);
	m_local_frontierpoint_markers.markers.resize(0);
}

void VizHelper::removeUnreachablePointMarkers()
{
	visualization_msgs::MarkerArray ftmarkers_old = m_unreachable_markers ;
	for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
		ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
	m_marker_unreachpointPub.publish(ftmarkers_old);
	m_unreachable_markers.markers.resize(0);
}

void VizHelper::setGlobalFrontierRegionMarkers(const vector<geometry_msgs::Point32>& globalfr_w, visualization_msgs::Marker& vizfr_markers )
{
	// reinit FR region
	for( int ii=0; ii < globalfr_w.size(); ii++ )
	{
		geometry_msgs::Point point_w;
		point_w.x = globalfr_w[ii].x ;
		point_w.y = globalfr_w[ii].y ;
		vizfr_markers.points.push_back( point_w ) ;
	}
}

void VizHelper::setGlobalFrontierPointMarkers( const geometry_msgs::Point32& target_goal, const vector<geometry_msgs::Point32>& global_fpts )
{
	for( int ii=0; ii < global_fpts.size(); ii++ )
	{
		float fx_g = global_fpts[ii].x ;
		float fy_g = global_fpts[ii].y ;
		float fdist_sq = (target_goal.x - fx_g) * (target_goal.x - fx_g) + (target_goal.y - fy_g) * (target_goal.y - fy_g) ;
		float fdist    = sqrt(fdist_sq);
		if( fdist < 0.0005 ) // don't create the marker that is potentially the target point
			continue ;

		visualization_msgs::Marker vizmarker = SetVizMarker( mn_global_FrontierID, visualization_msgs::Marker::ADD, fx_g, fy_g, 0.f,
				m_worldFrameId, 0.f, 1.f, 1.f, 0.5f, (float)FRONTIER_MARKER_SIZE );
		m_global_frontierpoint_markers.markers.push_back(vizmarker);
		mn_global_FrontierID++ ;
	}
}

void VizHelper::setLocalFrontierPointMarkers( const geometry_msgs::Point32& target_goal, const vector<geometry_msgs::Point32>& local_fpts )
{
	visualization_msgs::Marker fpts_markers;

	for( int ii=0; ii < local_fpts.size(); ii++ )
	{
		float fx_g = local_fpts[ii].x ;
		float fy_g = local_fpts[ii].y ;
		float fdist_sq = (target_goal.x - fx_g) * (target_goal.x - fx_g) + (target_goal.y - fy_g) * (target_goal.y - fy_g) ;
		float fdist    = sqrt(fdist_sq);
		if( fdist < 0.0005 ) // don't create the marker that is potentially the target point (within 10mm)
			continue ;

		visualization_msgs::Marker vizmarker = SetVizMarker( mn_FrontierID, visualization_msgs::Marker::ADD, fx_g, fy_g, 0.f,
				m_worldFrameId, 0.f, 1.f, 0.f, 1.f, (float)FRONTIER_MARKER_SIZE );
		m_local_frontierpoint_markers.markers.push_back(vizmarker);
		mn_FrontierID++ ;
	}
}

void VizHelper::setUnreachbleMarkers( const vector<geometry_msgs::Point32>& unreachable_pt )
{
	for( int ii=0; ii < unreachable_pt.size(); ii++ )
	{
		visualization_msgs::Marker vizmarker = SetVizMarker( mn_UnreachableFptID, visualization_msgs::Marker::ADD, unreachable_pt[ii].x, unreachable_pt[ii].y, 0.f,
				m_worldFrameId, 1.f, 1.f, 0.f, 1.f, (float)UNREACHABLE_MARKER_SIZE );
		m_unreachable_markers.markers.push_back(vizmarker);
		mn_UnreachableFptID++ ;
	}
}

void VizHelper::publishGlobalFrontierRegionMarkers( const visualization_msgs::Marker&	vizfr_markers )
{
	m_markerfrontierregionPub.publish(vizfr_markers);
}

//void VizHelper::publishLocalFrontierRegionMarkers(  )
//{
//
//}

void VizHelper::publishGlobalFrontierPointMarkers( )
{
	m_markerglobalfrontierPub.publish(m_global_frontierpoint_markers);
}

void VizHelper::publishLocalFrontierPointMarkers(  )
{
	m_markerfrontierPub.publish(m_local_frontierpoint_markers);
}

void VizHelper::publishUnreachbleMarkers()
{
	m_marker_unreachpointPub.publish(m_unreachable_markers);
}

void VizHelper::publishActiveBoundLines( )
{
	visualization_msgs::Marker vizbound_lines ;
	setActiveBound( m_rpos_w.x, m_rpos_w.y, mn_vizbound_width, mn_vizbound_height, vizbound_lines ) ;
	m_active_boundPub.publish( vizbound_lines );
}


void VizHelper::publishGoalPointMarker( const geometry_msgs::Point32& targetgoal )
{
	m_targetgoal_marker.points.clear();
	m_targetgoal_marker = SetVizMarker( -1, visualization_msgs::Marker::ADD, targetgoal.x, targetgoal.y, 0.f,
			m_worldFrameId,	1.f, 0.f, 1.f, 1.f, (float)TARGET_MARKER_SIZE );
	m_makergoalPub.publish(m_targetgoal_marker); // for viz
}

void VizHelper::publishAll()
{
	const vector<geometry_msgs::Point32> globalfr_w			= m_vizdata.global_frontier_region_w ;
	const vector<geometry_msgs::Point32> global_fpts_w 		= m_vizdata.global_fpts_w ;
	const vector<geometry_msgs::Point32> local_fpts_w 		= m_vizdata.local_fpts_w ;
	const vector<geometry_msgs::Point32> unreachable_fpts_w	= m_vizdata.unreachable_fpts_w ;
	const geometry_msgs::Point32 robot_pos_w				= m_vizdata.robot_pose_w ;
	const geometry_msgs::Point32 targetgoal_w				= m_vizdata.target_goal_w ;

//ROS_INFO("global FR region pts size: %d \n", globalfr_w.size() );

	visualization_msgs::Marker vizfr_markers = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, 1.f, 0.f, 0.f, 1.f, 0.1 );
	vizfr_markers.type = visualization_msgs::Marker::POINTS;

	setGlobalFrontierRegionMarkers( globalfr_w, vizfr_markers ); // cyan  (0,255,255)
//		setLocalFrontierRegionMarkers(  localfr_w );
	removeGlobalFrontierPointMarkers() ;
	setGlobalFrontierPointMarkers(targetgoal_w, global_fpts_w ) ;
	removeLocalFrontierPointMarkers() ;
	setLocalFrontierPointMarkers (targetgoal_w, local_fpts_w ) ; // green (0,255,0)
	removeUnreachablePointMarkers( );
	setUnreachbleMarkers( unreachable_fpts_w );

	publishGlobalFrontierRegionMarkers( vizfr_markers );
	publishGlobalFrontierPointMarkers(  ) ;
	publishLocalFrontierPointMarkers(  ) ;
	publishUnreachbleMarkers( );
	publishActiveBoundLines( );
	publishGoalPointMarker(  targetgoal_w );
}

void VizHelper::vizCallback( const neuro_explorer::VizDataStamped::ConstPtr& msg )
{
	m_vizdata = *msg ;
	bool bviz_flag = m_vizdata.viz_flag.data ;
	m_rpos_w = m_vizdata.robot_pose_w ;

	if(bviz_flag == false) // publish delete messages
	{
		// delete markers
		ROS_WARN("Refreshing viz data \n");
		// delete markers
		removeGlobalFrontierPointMarkers() ;
		removeLocalFrontierPointMarkers() ;

		return;
	}
	else
	{
		publishAll();
	}
}

void VizHelper::doneCallback( const std_msgs::Bool::ConstPtr& msg  )
{
	if( msg->data  == true)
		mb_explorationisdone = true;
}


}

