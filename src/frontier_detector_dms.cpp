/*********************************************************************
Copyright 2022 The Ewha Womans University.
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

#include "frontier_detector_dms.hpp"

namespace autoexplorer
{

FrontierDetectorDMS::FrontierDetectorDMS(const ros::NodeHandle private_nh_, const ros::NodeHandle &nh_):
m_nh_private(private_nh_),
m_nh(nh_),
mn_globalcostmapidx(0), mn_numthreads(16),
mb_isinitmotion_completed(false),
mp_cost_translation_table(NULL),
mb_strict_unreachable_decision(true),
me_prev_exploration_state( SUCCEEDED ), mb_nbv_selected(false), //, mn_prev_nbv_posidx(-1)
mb_allow_unknown(true),
mn_mapcallcnt(0), mn_moverobotcnt(0),
mf_totalcallbacktime_msec(0.f), mf_totalmotiontime_msec(0.f),
mf_avgcallbacktime_msec(0.f), 	mf_avgmotiontime_msec(0.f),
mf_avg_fd_sessiontime_msec(0.f), mf_total_fd_sessiontime_msec(0.f),
mf_avg_astar_sessiontime_msec(0.f), mf_total_astar_sessiontime_msec(0.f),
mn_num_classes(8)
{
	m_ae_start_time = ros::WallTime::now();

	float fcostmap_conf_thr, fgridmap_conf_thr; // mf_unreachable_decision_bound ;
	int nweakcomp_threshold ;

	m_nh.getParam("/autoexplorer/debug_data_save_path", m_str_debugpath);
	m_nh.getParam("/autoexplorer/ne_report_file", mstr_report_filename);

	m_nh.param("/autoexplorer/costmap_conf_thr", fcostmap_conf_thr, 0.1f);
	m_nh.param("/autoexplorer/gridmap_conf_thr", fgridmap_conf_thr, 0.8f);
	m_nh.param("/autoexplorer/occupancy_thr", mn_occupancy_thr, 50);
	m_nh.param("/autoexplorer/lethal_cost_thr", mn_lethal_cost_thr, 80);
	m_nh.param("/autoexplorer/global_width",  mn_globalmap_width, 	2048) ;
	m_nh.param("/autoexplorer/global_height", mn_globalmap_height, 	2048) ;
	m_nh.param("/autoexplorer/active_width",  mn_activemap_width, 	1024) ;
	m_nh.param("/autoexplorer/active_height", mn_activemap_height, 	1024) ;

	m_nh.param("/autoexplorer/unreachable_decision_bound", mf_neighoringpt_decisionbound, 0.2f);
	m_nh.param("/autoexplorer/weak_comp_thr", nweakcomp_threshold, 10);
	m_nh.param("/autoexplorer/num_downsamples", mn_numpyrdownsample, 0);
	m_nh.param("/autoexplorer/frame_id", m_worldFrameId, std::string("map"));
	m_nh.param("/autoexplorer/strict_unreachable_decision", mb_strict_unreachable_decision, true);
	m_nh.param("/autoexplorer/allow_unknown", mb_allow_unknown, true);
	m_nh.param("/move_base/global_costmap/resolution", mf_resolution, 0.05f) ;
	m_nh.param("move_base/global_costmap/robot_radius", mf_robot_radius, 0.12); // 0.3 for fetch

	m_nh.getParam("/tf_loader/fd_model_filepath", m_str_fd_modelfilepath);
	m_nh.getParam("/tf_loader/astar_model_filepath", m_str_astar_modelfilepath);
	m_nh.getParam("/tf_loader/covrew_model_filepath", m_str_covrew_modelfilepath);
	m_nh.param("/tf_loader/num_classes", mn_num_classes, 8);
	m_nh.param("/tf_loader/cnn_width", mn_cnn_width, 512);
	m_nh.param("/tf_loader/cnn_height", mn_cnn_height, 512);
ROS_INFO("global map size (%d %d) and dnn input size (%d %d): \n", mn_globalmap_height, mn_globalmap_width, mn_cnn_height, mn_cnn_width);

	mn_scale = pow(2, mn_numpyrdownsample);
	m_frontiers_region_thr = nweakcomp_threshold / mn_scale ;
	mn_roi_size = static_cast<int>( round( mf_robot_radius / mf_resolution ) ) * 2 ; // we never downsample costmap !!! dont scale it with roisize !!

	mn_globalmap_centx = mn_globalmap_width  / 2 ;
	mn_globalmap_centy = mn_globalmap_height / 2 ;

	//m_targetspub = m_nh.advertise<geometry_msgs::PointStamped>("detected_points", 10);
	m_currentgoalPub = m_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("curr_goalpose", 10);
	m_makergoalPub = m_nh.advertise<visualization_msgs::Marker>("curr_goal_shape",10);
	m_markercandPub = m_nh.advertise<visualization_msgs::MarkerArray>("detected_shapes", 10);
	m_markerfrontierPub = m_nh.advertise<visualization_msgs::MarkerArray>("filtered_shapes", 10);
	m_markerfrontierregionPub = m_nh.advertise<visualization_msgs::Marker>("FR_shapes", 1);
	m_marker_optcov_regionPub	  = m_nh.advertise<visualization_msgs::Marker>("cov_opt_shapes",1);
	m_marker_optastar_regionPub	  = m_nh.advertise<visualization_msgs::Marker>("astar_opt_shapes",1);
	m_marker_optensembled_regionPub = m_nh.advertise<visualization_msgs::Marker>("ensembled_opt_shapes",1);
	m_active_boundPub				= m_nh.advertise<visualization_msgs::Marker>("active_bound", 1);

	m_marker_unreachpointPub = m_nh.advertise<visualization_msgs::MarkerArray>("unreachable_shapes", 10);
	m_unreachpointPub 		 = m_nh.advertise<geometry_msgs::PoseStamped>("unreachable_posestamped",10);

	m_velPub		= m_nh.advertise<geometry_msgs::Twist>("cmd_vel",10);
	m_donePub		= m_nh.advertise<std_msgs::Bool>("exploration_is_done",1);
	m_startmsgPub	= m_nh.advertise<std_msgs::Bool>("begin_exploration",1);
	m_otherfrontierptsPub = m_nh.advertise<nav_msgs::Path>("goal_exclusive_frontierpoints_list",1);

//	m_mapframedataSub  	= m_nh.subscribe("map", 1, &FrontierDetectorDMS::mapdataCallback, this); // kmHan
	m_mapframedataSub  	= m_nh.subscribe("move_base/global_costmap/costmap", 1, &FrontierDetectorDMS::mapdataCallback, this); // kmHan
	m_currGoalSub 		= m_nh.subscribe("curr_goalpose",1 , &FrontierDetectorDMS::moveRobotCallback, this) ; // kmHan
	//m_globalCostmapSub 	= m_nh.subscribe("move_base/global_costmap/costmap", 1, &FrontierDetectorDMS::globalCostmapCallBack, this );

	m_poseSub		   	= m_nh.subscribe("pose", 10, &FrontierDetectorDMS::robotPoseCallBack, this);
	m_velSub			= m_nh.subscribe("cmd_vel", 10, &FrontierDetectorDMS::robotVelCallBack, this);
	m_unreachablefrontierSub = m_nh.subscribe("unreachable_posestamped", 1, &FrontierDetectorDMS::unreachablefrontierCallback, this);
	m_makeplan_client = m_nh.serviceClient<nav_msgs::GetPlan>("move_base/make_plan");

	mcvu_globalmapimg   = cv::Mat(mn_globalmap_height, mn_globalmap_width, CV_8U, cv::Scalar(127));
	mcvu_globalfrimg_ds = cv::Mat::zeros(mn_globalmap_height/2, mn_globalmap_width/2, CV_8U);
	mcvu_costmapimg     = cv::Mat(mn_globalmap_height, mn_globalmap_width, CV_8U, cv::Scalar(255));

	int ncostmap_roi_size = mn_roi_size ; // / 2 ;
	int ngridmap_roi_size = mn_roi_size ;
	mn_correctionwindow_width = mn_scale * 2 + 1 ; // the size of the correction search window

	mo_frontierfilter = FrontierFilter(
			ncostmap_roi_size, ngridmap_roi_size, m_str_debugpath, mn_numpyrdownsample,
			fgridmap_conf_thr, fcostmap_conf_thr, mn_occupancy_thr, mn_lethal_cost_thr,
			mn_globalmap_width, mn_globalmap_height,
			mf_resolution, mf_neighoringpt_decisionbound);

	//m_oFrontierFilter.SetUnreachableDistThr( funreachable_decision_bound ) ;

	while(!m_move_client.waitForServer(ros::Duration(5.0)))
	{
		ROS_INFO("Waiting for the move_base action server to come up");
	}
	ROS_INFO("move_base action server is up");

	mpo_costmap = new costmap_2d::Costmap2D();
	if (mp_cost_translation_table == NULL)
	{
		mp_cost_translation_table = new uint8_t[101];

		// special values:
		mp_cost_translation_table[0] = 0;  // NO obstacle
		mp_cost_translation_table[99] = 253;  // INSCRIBED obstacle
		mp_cost_translation_table[100] = 254;  // LETHAL obstacle
//		mp_cost_translation_table[-1] = 255;  // UNKNOWN

		// regular cost values scale the range 1 to 252 (inclusive) to fit
		// into 1 to 98 (inclusive).
		for (int i = 1; i < 99; i++)
		{
			mp_cost_translation_table[ i ] = uint8_t( ((i-1)*251 -1 )/97+1 );
		}
	}

// load init robot pose
	m_init_robot_pose = GetCurrRobotPose();

	// Set markers

	m_prev_frontier_set = set<pointset>();
	m_curr_frontier_set = set<pointset>();
	//m_exploration_goal = SetVizMarker( m_worldFrameId, 1.f, 0.f, 1.f, 0.5  );
	mn_FrontierID = 1;
	mn_UnreachableFptID = 0;
	ROS_INFO("autoexplorer has initialized with (%d) downsmapling map \n", mn_numpyrdownsample);

	std_msgs::Bool begin_task;
	begin_task.data = true;
	m_startmsgPub.publish( begin_task );

	m_last_oscillation_reset = ros::Time::now();

	// open ae report file
	string str_ae_report_file = m_str_debugpath + "/" + mstr_report_filename ;
ROS_INFO("report file: %s\n", str_ae_report_file.c_str());
	m_ofs_ae_report = ofstream( str_ae_report_file );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tf FD model
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	initalize_fd_model();
ROS_INFO("completed initializing fd model \n");
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tf astar model
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	initalize_astar_model();
ROS_INFO("completed initializing astar model \n");
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tf covrew model
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	initalize_covrew_model();
ROS_INFO("completed initializing cov rew model \n");
	// data handler
	m_data_handler = ImageDataHandler( mn_cnn_height, mn_cnn_width, 0.5f );
ROS_INFO("model num classes: %d \n", mn_num_classes);

}

FrontierDetectorDMS::~FrontierDetectorDMS()
{
	delete [] mp_cost_translation_table;
//Free tensorflow instances
	free(mpptf_fd_input_values);
	free(mpptf_fd_output_values);
	free(mptf_fd_output);
	free(mptf_fd_input);
	delete [] mpf_fd_data ;
	free(mpptf_astar_input_values);
	free(mpptf_astar_output_values);
	free(mptf_astar_output);
	free(mptf_astar_input);
	delete [] mpf_astar_data ;
}

void FrontierDetectorDMS::initalize_fd_model( )
{
	TF_Graph* ptf_fd_Graph = TF_NewGraph();
    mptf_fd_Status = TF_NewStatus();
    mptf_fd_SessionOpts = TF_NewSessionOptions();
    mptf_fd_RunOpts = NULL;

    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
    mptf_fd_Session = TF_LoadSessionFromSavedModel(mptf_fd_SessionOpts, mptf_fd_RunOpts, m_str_fd_modelfilepath.c_str(), &tags, ntags, ptf_fd_Graph, NULL, mptf_fd_Status);
    if(TF_GetCode(mptf_fd_Status) == TF_OK)
    {
        printf("TF_LoadSessionFrom FD SavedModel OK at the init state\n");
    }
    else
    {
        printf("%s",TF_Message(mptf_fd_Status));
    }

    //****** Get input tensor
    //TODO : need to use saved_model_cli to read saved_model arch
    int NumInputs = 1;
    mptf_fd_input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    mtf_fd_t0 = {TF_GraphOperationByName(ptf_fd_Graph, "serving_default_input_1"), 0};
    if(mtf_fd_t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
	printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

    mptf_fd_input[0] = mtf_fd_t0;

    //********* Get Output tensor
    int NumOutputs = 1;
    mptf_fd_output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    mtf_fd_t2 = {TF_GraphOperationByName(ptf_fd_Graph, "StatefulPartitionedCall"), 0};
    if(mtf_fd_t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    mptf_fd_output[0] = mtf_fd_t2;

    //*** allocate data for inputs & outputs
    mpptf_fd_input_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    mpptf_fd_output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    //*** data allocation
    mpf_fd_data = new float[ (mn_cnn_height) * (mn_cnn_width) ];
}
void FrontierDetectorDMS::initalize_astar_model( )
{
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
	int NumInputs = 1;
	mptf_astar_Graph = TF_NewGraph();
	mptf_astar_Status = TF_NewStatus();
	mptf_astar_SessionOpts = TF_NewSessionOptions();
	mptf_astar_RunOpts = NULL;

	mptf_astar_Session = TF_LoadSessionFromSavedModel(mptf_astar_SessionOpts, mptf_astar_RunOpts, m_str_astar_modelfilepath.c_str(), &tags, ntags, mptf_astar_Graph, NULL, mptf_astar_Status);
	if(TF_GetCode(mptf_astar_Status) == TF_OK)
	{
		printf("TF_LoadSessionFromSavedModel OK at the init state\n");
	}
	else
	{
		printf("%s",TF_Message(mptf_astar_Status));
	}

	//****** Get input tensor
	//TODO : need to use saved_model_cli to read saved_model arch
	mptf_astar_input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
	mtf_astar_t0 = {TF_GraphOperationByName(mptf_astar_Graph, "serving_default_input_1"), 0};
	if(mtf_astar_t0.oper == NULL)
		printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
	else
	printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

	mptf_astar_input[0] = mtf_astar_t0;

	//********* Get Output tensor
    int NumOutputs = 1;
	mptf_astar_output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
	mtf_astar_t2 = {TF_GraphOperationByName(mptf_astar_Graph, "StatefulPartitionedCall"), 0};
	if(mtf_astar_t2.oper == NULL)
		printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
	else
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

	mptf_astar_output[0] = mtf_astar_t2;

	//*** allocate data for inputs & outputs
	mpptf_astar_input_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
	mpptf_astar_output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

	//*** data allocation
	mpf_astar_data = new float[ mn_cnn_height * mn_cnn_width * 3 ];
}
void FrontierDetectorDMS::initalize_covrew_model( )
{
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
	int NumInputs = 1;
	TF_Graph* ptf_covrew_Graph = TF_NewGraph();

	mptf_covrew_Status = TF_NewStatus();
	mptf_covrew_SessionOpts = TF_NewSessionOptions();
	mptf_covrew_RunOpts = NULL;

	mptf_covrew_Session = TF_LoadSessionFromSavedModel(mptf_covrew_SessionOpts, mptf_covrew_RunOpts, m_str_covrew_modelfilepath.c_str(), &tags, ntags, ptf_covrew_Graph, NULL, mptf_covrew_Status);
	if(TF_GetCode(mptf_covrew_Status) == TF_OK)
	{
		printf("TF_LoadSessionFrom CovRew SavedModel OK at the init state\n");
	}
	else
	{
		printf("%s",TF_Message(mptf_covrew_Status));
	}

	//****** Get input tensor
	//TODO : need to use saved_model_cli to read saved_model arch
	mptf_covrew_input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
	mtf_covrew_t0 = {TF_GraphOperationByName(ptf_covrew_Graph, "serving_default_input_1"), 0};
	if(mtf_covrew_t0.oper == NULL)
		printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
	else
	printf("TF_GraphOperation covrew serving_default_input_1 is OK\n");

	mptf_covrew_input[0] = mtf_covrew_t0;

	//********* Get Output tensor
    int NumOutputs = 1;
	mptf_covrew_output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
	mtf_covrew_t2 = {TF_GraphOperationByName(ptf_covrew_Graph, "StatefulPartitionedCall"), 0};
	if(mtf_covrew_t2.oper == NULL)
		printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
	else
	printf("TF_GraphOperation CovRew StatefulPartitionedCall is OK\n");

	mptf_covrew_output[0] = mtf_covrew_t2;

	//*** allocate data for inputs & outputs
	mpptf_covrew_input_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
	mpptf_covrew_output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

	//*** data allocation
	mpf_covrew_data = new float[ mn_cnn_height * mn_cnn_width * 1 ]; // input_map is 3ch 512 x 512
//	mpf_covrew_result = new float[ mn_processed_gmap_height * mn_processed_gmap_width * 3 ];
}


void FrontierDetectorDMS::initmotion(  const float& fvx = 0.f, const float& fvy = 0.f, const float& ftheta = 1.f  )
{
ROS_INFO("+++++++++++++++++ Start the init motion ++++++++++++++\n");
	geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = fvx;
    cmd_vel.linear.y = fvy;
    cmd_vel.angular.z = ftheta;

    uint32_t start_time = ros::Time::now().sec ;
    uint32_t curr_time = start_time ;
    while( curr_time < start_time + 4 )
    {
		m_velPub.publish(cmd_vel);
		curr_time = ros::Time::now().sec ;
		ros::Duration(0.1).sleep();
    }

	cmd_vel.angular.z = 0.0;
	m_velPub.publish(cmd_vel);
ROS_INFO("+++++++++++++++++ end of the init motion ++++++++++++++\n");
}


cv::Point2f FrontierDetectorDMS::gridmap2world( cv::Point img_pt_roi  )
{
	float fgx =  static_cast<float>(img_pt_roi.x) * m_gridmap.info.resolution + m_gridmap.info.origin.position.x  ;
	float fgy =  static_cast<float>(img_pt_roi.y) * m_gridmap.info.resolution + m_gridmap.info.origin.position.y  ;

	return cv::Point2f( fgx, fgy );
}

cv::Point FrontierDetectorDMS::world2gridmap( cv::Point2f grid_pt)
{
	float fx = (grid_pt.x - m_gridmap.info.origin.position.x) / m_gridmap.info.resolution ;
	float fy = (grid_pt.y - m_gridmap.info.origin.position.y) / m_gridmap.info.resolution ;

	return cv::Point( (int)fx, (int)fy );
}

void FrontierDetectorDMS::generateGridmapFromCostmap( )
{
	m_gridmap = m_globalcostmap ;
	m_gridmap.info = m_globalcostmap.info ;

	int gmheight = m_globalcostmap.info.height ;
	int gmwidth = m_globalcostmap.info.width ;

	for( int ii =0 ; ii < gmheight; ii++)
	{
		for( int jj = 0; jj < gmwidth; jj++)
		{
			int8_t obs_cost  = m_globalcostmap.data[ ii * gmwidth + jj] ;

			if ( obs_cost < 0) // if unknown
			{
				m_gridmap.data[ ii*gmwidth + jj ] = -1 ;
			}
			else if( obs_cost > 97 ) // mp_cost_translation_table[51:98] : 130~252 : possibly circumscribed ~ inscribed
			{
				m_gridmap.data[ ii*gmwidth + jj] = 100 ;
			}
			else
			{
				m_gridmap.data[ ii*gmwidth + jj] = 0 ;
			}
		}
	}
	//ROS_INFO("cmap: ox oy: %f %f \n W H: %d %d", m_globalcostmap.info.origin.position.x, m_globalcostmap.info.origin.position.y, m_globalcostmap.info.width, m_globalcostmap.info.height);
	//ROS_INFO("gmap: ox oy: %f %f \n W H: %d %d", m_gridmap.info.origin.position.x, m_gridmap.info.origin.position.y, m_gridmap.info.width, m_gridmap.info.height);
}

void FrontierDetectorDMS::setVizMarkerFromPointClass( const PointClassSet& pointset, visualization_msgs::Marker& vizmarker, const rgb& init_color, float fsize = 0.1 )
{
	vizmarker = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, init_color.r, init_color.g, init_color.b, fsize);
	vizmarker.type = visualization_msgs::Marker::POINTS;

	std::vector<PointClass> pts_class = pointset.GetPointClass() ;
	std::vector<rgb> pts_rgb = pointset.GetPointColor() ;
	std_msgs::ColorRGBA color ;
	for( int ii=0; ii < pts_class.size(); ii++ )
	{
		//PointClass point = points[ii];
		rgb c = pts_rgb[ii];
		FrontierPoint oFRpoint( cv::Point(pts_class[ii].x, pts_class[ii].y), m_gridmap.info.height, m_gridmap.info.width,
								m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x, m_gridmap.info.resolution, mn_numpyrdownsample );
		geometry_msgs::Point point_w;
		point_w.x = oFRpoint.GetInitWorldPosition().x ;
		point_w.y = oFRpoint.GetInitWorldPosition().y ;
		color.r = c.r;
		color.g = c.g;
		color.b = c.b;
		color.a = 1.0;
		vizmarker.colors.push_back( color ) ;
		vizmarker.points.push_back( point_w ) ;
	}
}

void FrontierDetectorDMS::setActiveBound( const float& frx_w, const float& fry_w, const int& ngmwidth, const int& ngmheight, visualization_msgs::Marker& vizmarker )
{
	cv::Point2f fbound_size_world = gridmap2world( cv::Point( ngmwidth, ngmheight ) );

	float fox_w = frx_w - fbound_size_world.x / 2 ;
	float foy_w = fry_w - fbound_size_world.y / 2 ;
	float fBx_w = fox_w + fbound_size_world.x ;
	float fBy_w = foy_w + fbound_size_world.y ;

	vizmarker = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, 1.f, 0.65, 0.f, 0.2);
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

void FrontierDetectorDMS::publishDoneExploration( )
{
ROS_INFO("writing a final reports \n");
	ros::WallTime ae_end_time = ros::WallTime::now();
	double total_exploration_time = (ae_end_time - m_ae_start_time ).toNSec() * 1e-6;

	// time analysis
	// Note that DNN take a large amount of time for the 1st run...

	double fr_detection_time_fullavg, fr_detection_time_runavg ;
	double astar_time_fullavg, astar_time_runavg ;
	double covrew_time_fullavg, covrew_time_runavg ;
	double fmapcallcnt = (double)(mn_mapcallcnt) ;
	double fmoverobotcnt = (double)(mn_moverobotcnt);

	fr_detection_time_runavg = std::accumulate(mvf_fr_detection_time.begin()+1, mvf_fr_detection_time.end(), 0.0) / (fmapcallcnt-1) ;
	fr_detection_time_fullavg = std::accumulate(mvf_fr_detection_time.begin(), mvf_fr_detection_time.end(), 0.0) / (fmapcallcnt) ;

	astar_time_runavg = std::accumulate(mvf_astar_time.begin()+1, mvf_astar_time.end(), 0.0) / (fmapcallcnt-1.0) ;
	astar_time_fullavg = std::accumulate(mvf_astar_time.begin(), mvf_astar_time.end(), 0.0) /  (fmapcallcnt) ;

	covrew_time_runavg = std::accumulate(mvf_covrew_time.begin()+1, mvf_covrew_time.end(), 0.0) / (fmapcallcnt-1.0) ;
	covrew_time_fullavg = std::accumulate(mvf_covrew_time.begin(), mvf_covrew_time.end(), 0.0) / (fmapcallcnt) ;

	m_ofs_ae_report << "******************* exploration report ****************************" << '\n';
	m_ofs_ae_report<< std::fixed << std::setprecision(2) ;
	m_ofs_ae_report << "total num of mapcallback         : " << mn_mapcallcnt << '\n';
	m_ofs_ae_report << "total num of move robot cnt      : " << mn_moverobotcnt << '\n';
	m_ofs_ae_report << "tot / avg callback time     (ms) : " << mf_totalcallbacktime_msec << "\t" << mf_totalcallbacktime_msec / fmapcallcnt << '\n';
	m_ofs_ae_report << "tot / avg motion time       (s)  : " << mf_totalmotiontime_msec / 1000.0 << "\t" << mf_totalmotiontime_msec / (fmoverobotcnt * 1000.0) << '\n';
	m_ofs_ae_report << "total exploration time      (s)  : " << total_exploration_time / 1000.0 << '\n';
	m_ofs_ae_report << '\n';

	m_ofs_ae_report << setw(10) << "model  name"  << setw(35) << "full avg time (ms)"           << setw(35) << "except the 1st call (ms)"    << "\n";
	m_ofs_ae_report << setw(10) << "fr dete net"  << setw(35) << fr_detection_time_fullavg 		<< setw(35) << fr_detection_time_runavg << "\n";
	m_ofs_ae_report << setw(10) << "astar   net"  << setw(35) << astar_time_fullavg 			<< setw(35) << astar_time_runavg 		<< "\n";
	m_ofs_ae_report << setw(10) << "cov rew net"  << setw(35) << covrew_time_fullavg 			<< setw(35) << covrew_time_runavg 		<< "\n";

	m_ofs_ae_report << "\n";

	m_ofs_ae_report << "******************* individual net report ************************" << endl;
	m_ofs_ae_report << setw(7) << "Call No." << setw(25) << "fr net time (ms)"       << setw(25)  << "astar net time (ms)"      << setw(25) << "cov rew net time (ms)"    << endl;
for(int idx=0; idx < mvf_fr_detection_time.size(); idx++)
{
	m_ofs_ae_report << setw(7) << idx        << setw(25) << mvf_fr_detection_time[idx] <<  setw(25) << mvf_astar_time[idx] << setw(25) << mvf_covrew_time[idx] << endl;
}

m_ofs_ae_report.close();

//	ROS_INFO("total map data callback counts : %d \n", mn_mapcallcnt  );
//	ROS_INFO("total callback time (sec) %f \n", mf_totalcallbacktime_msec / 1000 );
//	ROS_INFO("total planning time (sec) %f \n", mf_totalplanningtime_msec / 1000 );
//	ROS_INFO("avg callback time (msec) %f \n", favg_callback_time  );
//	ROS_INFO("avg planning time (msec) %f \n", favg_planning_time  );

	ROS_INFO("The exploration task is done... publishing -done- msg" );
	std_msgs::Bool done_task;
	done_task.data = true;

	m_frontierpoint_markers = visualization_msgs::MarkerArray() ;
	m_markerfrontierPub.publish(m_frontierpoint_markers);
	m_targetgoal_marker = visualization_msgs::Marker() ;
	m_makergoalPub.publish(m_targetgoal_marker); // for viz

	m_donePub.publish( done_task );

	ros::spinOnce();
}

void FrontierDetectorDMS::publishFrontierPointMarkers()
{
	visualization_msgs::MarkerArray ftmarkers_old = m_frontierpoint_markers ;
	for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
		ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
	m_markerfrontierPub.publish(ftmarkers_old);
	m_frontierpoint_markers.markers.resize(0);

	// publish fpts to Rviz
	for (const auto & pi : m_curr_frontier_set)
	{
		visualization_msgs::Marker vizmarker = SetVizMarker( mn_FrontierID, visualization_msgs::Marker::ADD, pi.p[0], pi.p[1], 0.f, m_worldFrameId, 0.f, 1.f, 0.f, (float)FRONTIER_MARKER_SIZE );
		m_frontierpoint_markers.markers.push_back(vizmarker);
		mn_FrontierID++ ;
	}
	m_markerfrontierPub.publish(m_frontierpoint_markers);
}

void FrontierDetectorDMS::publishFrontierRegionMarkers( const visualization_msgs::Marker& vizfrontier_regions  )
{
	// publish fpt regions to Rviz
	m_markerfrontierregionPub.publish(vizfrontier_regions);
}

void FrontierDetectorDMS::publishOptCovRegionMarkers( const visualization_msgs::Marker& vizcovopt_regions  )
{
	m_marker_optcov_regionPub.publish(vizcovopt_regions);
}

void FrontierDetectorDMS::publishOptAstarRegionMarkers( const visualization_msgs::Marker& vizastaropt_regions  )
{
	m_marker_optastar_regionPub.publish(vizastaropt_regions);
}

void FrontierDetectorDMS::publishOptEnsembledRegionMarkers( const visualization_msgs::Marker& vizopt_regions  )
{
	m_marker_optensembled_regionPub.publish(vizopt_regions);
}

void FrontierDetectorDMS::publishActiveBoundLines( const visualization_msgs::Marker& vizbound_lines )
{
	m_active_boundPub.publish( vizbound_lines );
}


void FrontierDetectorDMS::publishGoalPointMarker( const geometry_msgs::PoseWithCovarianceStamped& targetgoal )
{
	m_targetgoal_marker.points.clear();
	m_targetgoal_marker = SetVizMarker( -1, visualization_msgs::Marker::ADD, targetgoal.pose.pose.position.x, targetgoal.pose.pose.position.y, 0.f,
			m_worldFrameId,	1.f, 0.f, 1.f, (float)TARGET_MARKER_SIZE );
	m_makergoalPub.publish(m_targetgoal_marker); // for viz
}

void FrontierDetectorDMS::publishUnreachableMarkers( ) //const geometry_msgs::PoseStamped& unreachablepose )
{
	// first we refresh/update viz markers
	visualization_msgs::MarkerArray ftmarkers_old = m_unreachable_markers ;
	for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
		ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
	m_marker_unreachpointPub.publish(ftmarkers_old);
	m_unreachable_markers.markers.resize(0);

	const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set) ;
	{
		// create new markers and publish them to Rviz
		for (const auto & pi : m_unreachable_frontier_set)
		{
			visualization_msgs::Marker vizmarker = SetVizMarker( mn_UnreachableFptID, visualization_msgs::Marker::ADD, pi.p[0], pi.p[1], 0.f, m_worldFrameId, 1.f, 1.f, 0.f, (float)UNREACHABLE_MARKER_SIZE);
			m_unreachable_markers.markers.push_back(vizmarker);
			mn_UnreachableFptID++ ;
		}
	}
	m_marker_unreachpointPub.publish(m_unreachable_markers);
}

void FrontierDetectorDMS::appendUnreachablePoint( const geometry_msgs::PoseStamped& unreachablepose )
{
	pointset ufpt(unreachablepose.pose.position.x, unreachablepose.pose.position.y );
	// append the incoming unreachable fpt to the list
	if( !mb_strict_unreachable_decision)
	{
		// this case, we only check the frontier sanity check... don't care about physical path plan to the target.
		// if still valid frontier, we don't register this point to unreachable list b/c we think MoveBase() can guide the robot to the goal. Yet, Robot will stuck if this is not the case.
		nav_msgs::OccupancyGrid globalcostmap;
		std::vector<signed char> cmdata;
		{
			const std::unique_lock<mutex> lock(mutex_costmap);
			globalcostmap = m_globalcostmap;
			cmdata = globalcostmap.data;
		}
		int ncmx = static_cast<int>( (ufpt.p[0] - globalcostmap.info.origin.position.x) / globalcostmap.info.resolution ) ;
		int ncmy = static_cast<int>( (ufpt.p[1] - globalcostmap.info.origin.position.y) / globalcostmap.info.resolution ) ;
		//if( frontier_sanity_check(ncmx, ncmy, globalcostmap.info.width, cmdata) )
		if( cmdata[ ncmy * globalcostmap.info.width + ncmx ] == -1 )
			return ;
	}

	{
		const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set) ;
		m_unreachable_frontier_set.insert( ufpt ) ;
		ROS_WARN("@unreachablefrontierCallback Registering (%f %f) as the unreachable pt. Tot unreachable pts: %d \n", ufpt.p[0], ufpt.p[1], m_unreachable_frontier_set.size() );
	}

}

void FrontierDetectorDMS::updatePrevFrontierPointsList( )
{
	{
		const std::unique_lock<mutex> lock_curr(mutex_curr_frontier_set);
		const std::unique_lock<mutex> lock_prev(mutex_prev_frontier_set);
		set<pointset> frontier_set_tmp ;
		frontier_set_tmp = m_curr_frontier_set ;
		m_prev_frontier_set.clear() ;
		m_curr_frontier_set.clear() ;
		m_prev_frontier_set = frontier_set_tmp;
	}
}

void FrontierDetectorDMS::initGlobalmapimgs( const int& cmheight, const int& cmwidth, const nav_msgs::OccupancyGrid& globalcostmap )
{
	for( int ii =0 ; ii < cmheight; ii++)
	{
		for( int jj = 0; jj < cmwidth; jj++)
		{
			int8_t occupancy = m_gridmap.data[ ii * cmwidth + jj ]; // dynamic gridmap size
			int8_t obs_cost  = globalcostmap.data[ ii * cmwidth + jj] ;
			int y_ = (mn_roi_origy + ii) ;
			int x_ = (mn_roi_origx + jj) ;

			if ( occupancy < 0 && obs_cost < 0)
			{
				mcvu_globalmapimg.data[ y_ * mn_globalmap_width + x_ ] = static_cast<uchar>(dffp::MapStatus::UNKNOWN) ;
			}
			else if( occupancy >= 0 && occupancy < mn_occupancy_thr && obs_cost < 98) // mp_cost_translation_table[51:98] : 130~252 : possibly circumscribed ~ inscribed
			{
				mcvu_globalmapimg.data[ y_ * mn_globalmap_width + x_ ] = static_cast<uchar>(dffp::MapStatus::FREE) ;
			}
			else
			{
				mcvu_globalmapimg.data[ y_ * mn_globalmap_width + x_ ] = static_cast<uchar>(dffp::MapStatus::OCCUPIED) ;
			}
		}
	}

//ROS_INFO("mcvu_globalmapimg has been processed %d %d \n", mcvu_globalmapimg.rows, mcvu_globalmapimg.cols );

// process costmap
	for( int ii =0 ; ii < cmheight; ii++)
	{
		for( int jj = 0; jj < cmwidth; jj++)
		{
			int8_t val  = globalcostmap.data[ ii * cmwidth + jj] ;
			int y_ = (mn_roi_origy + ii) ;
			int x_ = (mn_roi_origx + jj) ;
			mcvu_costmapimg.data[ y_ * mn_globalmap_width + x_ ] = 	val < 0 ? 255 : mp_cost_translation_table[val];
		}
	}
}

void FrontierDetectorDMS::copyFRtoGlobalmapimg( const cv::Rect& roi_active, const cv::Mat& fr_img )
{
	// fr_img : 512 x 512
	// mcvu_globalfrimg_ds : 1024 x 1024
	int nwidth_ds 	= roi_active.width / 2  ;
	int nheight_ds	= roi_active.height/ 2 ;
	int nox_ds 		= roi_active.x / 2 ;
	int noy_ds 		= roi_active.y / 2 ;
	cv::Rect roi_active_ds( nox_ds, noy_ds, nwidth_ds, nheight_ds ); //cmwidth, cmheight );

	cv::Mat map_active ; // partial (on reconstructing map)
	map_active = mcvu_globalfrimg_ds( roi_active_ds );
	fr_img.copyTo(map_active);
}

int  FrontierDetectorDMS::locateFptsFromFRimg( const cv::Mat& cvFRimg, const int& nxoffset, const int& nyoffset, vector<FrontierPoint>& voFrontierCands   )
{
	vector<vector<cv::Point> > contours_plus_offset ;
	vector<cv::Vec4i> opt_hierarchy;
	cv::findContours( cvFRimg, contours_plus_offset, opt_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

	const int cmheight = m_gridmap.info.height ;
	const int cmwidth  = m_gridmap.info.width ;
	const int cmresolution = m_gridmap.info.resolution ;
	vector<signed char> gmdata = m_gridmap.data ;
	vector<signed char> cmdata = m_globalcostmap.data ;

	int num_valid_frontier_point = 0;
	if( contours_plus_offset.size() == 0 )
	{
		ROS_WARN("There is no opt FR region in the input map img\n");
		return 0;
	}
	else // We found opt fr in the current map. We append the new opt points to m_curr_frontier_set accordingly
	{
		geometry_msgs::Point point_w;
		vector<cv::Point> vecents_offset;  // shifted by the offet param
		for(int i=0; i < contours_plus_offset.size(); i++)
		{
			int nx =0, ny =0 ;
			int ncnt = 0 ;
			vector<cv::Point> fr_plus_offset = contours_plus_offset[i];
			for( int j=0; j < fr_plus_offset.size(); j++)
			{
				FrontierPoint oFRpoint( cv::Point(fr_plus_offset[j].x, fr_plus_offset[j].y), cmheight, cmwidth,
										m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x,
									   cmresolution, mn_numpyrdownsample );
				point_w.x = oFRpoint.GetInitWorldPosition().x ;
				point_w.y = oFRpoint.GetInitWorldPosition().y ;
	//			vizfrontier_regions.points.push_back( point_w ) ;
				nx += fr_plus_offset[j].x ;
				ny += fr_plus_offset[j].y ;
	//			ROS_INFO("%d %d / %d %d \n", contour_plus_offset[j].x, contour_plus_offset[j].y, img_plus_offset.rows, img_plus_offset.cols );
				ncnt += 1;
			}
			int ncx = nx / ncnt ;
			int ncy = ny / ncnt ;

			cv::Point ncent( ncx,  ncy ) ;
			vecents_offset.push_back( ncent );
		}
		CV_Assert( contours_plus_offset.size() == vecents_offset.size() );

		for( int i = 0; i < contours_plus_offset.size(); i++ )
		{
			vector<cv::Point> fr_plus_offset = contours_plus_offset[i] ;
			if(fr_plus_offset.size() < m_frontiers_region_thr ) // don't care about small frontier regions
				continue ;

			int ncentx_offset = vecents_offset[i].x ;
			int ncenty_offset = vecents_offset[i].y ;

			int nmindist = 10000000 ;
			int nmindistidx = -1;

			for (int j=0; j < fr_plus_offset.size(); j++)
			{
				int nx = fr_plus_offset[j].x ;
				int ny = fr_plus_offset[j].y ;
				int ndist = std::sqrt( (nx - ncentx_offset) * (nx - ncentx_offset) + (ny - ncenty_offset) * (ny - ncenty_offset) );
				if(ndist < nmindist)
				{
					nmindist = ndist ;
					nmindistidx = j ;
				}
			}

			CV_Assert(nmindistidx >= 0);
			cv::Point frontier_offset = fr_plus_offset[nmindistidx];

			//ROS_WARN(" %d %d \n", frontier_offset.x, frontier_offset.y);

			cv::Point frontier ;
			frontier.x = frontier_offset.x  ;
			frontier.y = frontier_offset.y  ;

			FrontierPoint oPoint( frontier, cmheight, cmwidth,
									m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x,
						   cmresolution, mn_numpyrdownsample );

	/////////////////////////////////////////////////////////////////////
	// 				We need to run position correction here
	/////////////////////////////////////////////////////////////////////
			cv::Point init_pt 		= oPoint.GetInitGridmapPosition() ; 	// position @ ds0 (original sized map)
			cv::Point corrected_pt	= oPoint.GetCorrectedGridmapPosition() ;
			correctFrontierPosition( m_gridmap, init_pt, mn_correctionwindow_width, corrected_pt  );

			oPoint.SetCorrectedCoordinate(corrected_pt);
			voFrontierCands.push_back(oPoint);
		}

		// run filter
		const float fcm_conf_thr = mo_frontierfilter.GetCostmapConf() ;
		const float fgm_conf_thr = mo_frontierfilter.GetGridmapConf() ;

		// eliminate frontier points at obtacles

		if( m_globalcostmap.info.width > 0 )
		{
			mo_frontierfilter.measureCostmapConfidence(m_globalcostmap, voFrontierCands);
//ROS_INFO("done CM filter \n");
			mo_frontierfilter.measureGridmapConfidence(m_gridmap, voFrontierCands);
//ROS_INFO("done GM filter \n");

			for(size_t idx=0; idx < voFrontierCands.size(); idx++)
			{
				cv::Point frontier_in_gm = voFrontierCands[idx].GetCorrectedGridmapPosition();
				bool bisfrontier = is_frontier_point(frontier_in_gm.x, frontier_in_gm.y, cmwidth, cmheight, gmdata );
				int gmidx = cmwidth * frontier_in_gm.y	+	frontier_in_gm.x ;
				bool bisexplored = cmdata[gmidx] >=0 ? true : false ;
				voFrontierCands[idx].SetFrontierFlag( fcm_conf_thr, fgm_conf_thr, bisexplored, bisfrontier );

//ROS_INFO("%f %f %d %d\n", voFrontierCands[idx].GetInitWorldPosition().x, voFrontierCands[idx].GetInitWorldPosition().y,
//						  voFrontierCands[idx].GetCorrectedGridmapPosition().x, voFrontierCands[idx].GetCorrectedGridmapPosition().y);

			}
			set<pointset> unreachable_frontiers;
			{
				const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set) ;
				unreachable_frontiers = m_unreachable_frontier_set ;
			}
			mo_frontierfilter.computeReachability( unreachable_frontiers, voFrontierCands );
		}

		{
			const std::unique_lock<mutex> lock(mutex_curr_frontier_set);
			for (size_t idx=0; idx < voFrontierCands.size(); idx++)
			{
//ROS_INFO("<%f %f> confidence  %d \n", voFrontierCands[idx].GetCorrectedWorldPosition().x, voFrontierCands[idx].GetCorrectedWorldPosition().y,
//		voFrontierCands[idx].isConfidentFrontierPoint() );
				if( voFrontierCands[idx].isConfidentFrontierPoint() )
				{
					//valid_frontier_indexs.push_back( idx );
					cv::Point2f frontier_in_world = voFrontierCands[idx].GetCorrectedWorldPosition();
					pointset pt( frontier_in_world.x, frontier_in_world.y );
					m_curr_frontier_set.insert( pt );
					num_valid_frontier_point++;
				}
			}
		}
	}

	return num_valid_frontier_point;
}

void FrontierDetectorDMS::globalCostmapCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
	//ROS_INFO("@globalCostmapCallBack \n");
	const std::unique_lock<mutex> lock(mutex_costmap);
//ROS_INFO("cm callback is called \n");
	m_globalcostmap = *msg ;
	mu_cmheight = m_globalcostmap.info.height ;
	mu_cmwidth = m_globalcostmap.info.width ;
}

void FrontierDetectorDMS::globalCostmapUpdateCallback(const map_msgs::OccupancyGridUpdate::ConstPtr& msg )
{
	const std::unique_lock<mutex> lock(mutex_costmap);
	map_msgs::OccupancyGridUpdate cm_updates = *msg;
	std::vector<signed char> Data=cm_updates.data;
	int x = cm_updates.x ;
	int y = cm_updates.y ;
	int height = cm_updates.height ;
	int width  = cm_updates.width ;
	int cmwidth = m_globalcostmap.info.width ;
	int dataidx = 0 ;
	{
		for(int ii=0; ii < height; ii++)
		{
			for(int jj=0; jj < width; jj++)
			{
				int idx = x + jj + ii * cmwidth ;
				m_globalcostmap.data[dataidx] = Data[dataidx] ;
				dataidx++;
			}
		}
	}
}


void FrontierDetectorDMS::robotPoseCallBack( const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg )
{
	m_robotpose = *msg ;
}

void FrontierDetectorDMS::robotVelCallBack( const geometry_msgs::Twist::ConstPtr& msg )
{
	m_robotvel = *msg ;
}

int FrontierDetectorDMS::saveMap( const nav_msgs::OccupancyGrid& map, const string& infofilename, const string& mapfilename )
{
	int nwidth  = map.info.width ;
	int nheight = map.info.height ;
	float fox = map.info.origin.position.x ;
	float foy = map.info.origin.position.y ;
	float fres = map.info.resolution ;
	std::vector<signed char> cmdata;
	cmdata = map.data ;

	if( nwidth == 0 || nheight == 0 )
	{
		ROS_ERROR("Cannot save incomplete costmap \n");
		return -1;
	}

	geometry_msgs::PoseStamped robotpose = GetCurrRobotPose();
	float frx_w = (robotpose.pose.position.x - m_gridmap.info.origin.position.x) / m_gridmap.info.resolution ;
	float fry_w = (robotpose.pose.position.y - m_gridmap.info.origin.position.y) / m_gridmap.info.resolution ;
	int nrx_g = static_cast<int>(frx_w) ;
	int nry_g = static_cast<int>(fry_w) ;
	//int8_t cost = cmdata[ nwidth * nry_g + nrx_g ] ;

	std::ofstream ofs_info( infofilename );
	std::ofstream ofs_map(mapfilename);

	std::vector<signed char> mapdata =map.data;
	ofs_info << nwidth << " " << nheight << " " << fox << " " << foy << " " << fres << " " <<
			robotpose.pose.position.x << " " << robotpose.pose.position.y << endl;

	for( int ii=0; ii < nheight; ii++ )
	{
		for( int jj=0; jj < nwidth; jj++ )
		{
			int dataidx = ii * nwidth + jj ;
			int val = static_cast<int>( mapdata[ dataidx ] ) ;
			ofs_map << val << " ";
		}
		ofs_map << "\n";
	}
	ofs_map.close();
	return 1;
}

// save prev
int FrontierDetectorDMS::saveFrontierPoints( const nav_msgs::OccupancyGrid& map, const nav_msgs::Path& msg_frontiers, int bestidx, const string& frontierfile  )
{

	int nwidth  = map.info.width ;
	int nheight = map.info.height ;
	float fox = map.info.origin.position.x ;
	float foy = map.info.origin.position.y ;
	float fres = map.info.resolution ;
	std::vector<signed char> cmdata;
	cmdata = map.data ;


	std::ofstream ofs_fpts( frontierfile );
	ofs_fpts << msg_frontiers.poses.size() << " " << bestidx << endl;
	{
		// append valid previous frontier points by sanity check
		for (const auto & pi : msg_frontiers.poses)
		{
			int ngmx = static_cast<int>( (pi.pose.position.x - fox) / fres ) ;
			int ngmy = static_cast<int>( (pi.pose.position.y - fox) / fres ) ;

			//int8_t cost = cmdata[ nwidth * ngmy + ngmx ] ;

			ofs_fpts << pi.pose.position.x << " " << pi.pose.position.y << " " << ngmx << " " << ngmy << "\n"; // << cost << "\n" ;
		}
	}

	ofs_fpts.close();

	return 1;
}

int FrontierDetectorDMS::savefrontiercands( const nav_msgs::OccupancyGrid& map, const vector<FrontierPoint>& voFrontierPoints, const string& frontierfile )
{

	int nwidth  = map.info.width ;
	int nheight = map.info.height ;
	float fox = map.info.origin.position.x ;
	float foy = map.info.origin.position.y ;
	float fres = map.info.resolution ;

	std::ofstream ofs_currfpts( frontierfile );

	for(const auto & fpt : voFrontierPoints)
	{
		ofs_currfpts <<
		fpt.GetCorrectedWorldPosition().x << " " << fpt.GetCorrectedWorldPosition().y << " " <<
		fpt.GetCorrectedGridmapPosition().x << " " << fpt.GetCorrectedGridmapPosition().y << " " <<
		fpt.GetCMConfidence() << " " << fpt.GetGMConfidence() << "\n" ;
	}
	ofs_currfpts.close();

	return 1;
}


int FrontierDetectorDMS::frontier_summary( const vector<FrontierPoint>& voFrontierCurrFrame )
{
	ROS_INFO("\n========================================================================== \n"
			  "============================Frontier Summary  =============================\n\n");
	ROS_INFO("\n\n frontier found in curr frame: \n\n");
	for(size_t idx=0; idx < voFrontierCurrFrame.size(); idx++)
	{
		FrontierPoint fpt= voFrontierCurrFrame[idx];
		if(fpt.isConfidentFrontierPoint())
		{
			ROS_INFO(" (%f %f) is (Reachablity: %s) (Frontierness: %s) \n",
				fpt.GetCorrectedWorldPosition().x, fpt.GetCorrectedWorldPosition().y, fpt.isReachable()?"true":"false", fpt.isConfidentFrontierPoint()?"true":"false" );
		}
		else
		{
			ROS_ERROR(" (%f %f) is (Reachablity: %s) (Frontierness: %s) \n",
				fpt.GetCorrectedWorldPosition().x, fpt.GetCorrectedWorldPosition().y, fpt.isReachable()?"true":"false", fpt.isConfidentFrontierPoint()?"true":"false" );
		}
	}

	ROS_INFO("\n\n Tot cumulated frontier points to process: \n\n");
	for (const auto & di : m_curr_frontier_set )
	{
			ROS_INFO("cumulated pts: (%f %f) \n", di.p[0], di.p[1]);
	}

	ROS_INFO("========================================================================== \n\n");

	return 1;
}


void FrontierDetectorDMS::updateUnreachablePointSet( const nav_msgs::OccupancyGrid& globalcostmap )
{
	std::vector<signed char> cmdata = globalcostmap.data ;
	{
		const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set) ;
		auto it = m_unreachable_frontier_set.begin() ;
		while (it != m_unreachable_frontier_set.end() )
		{
			auto it_element = it++;
			int ncmx = static_cast<int>( ((*it_element).p[0] - globalcostmap.info.origin.position.x) / globalcostmap.info.resolution ) ;
			int ncmy = static_cast<int>( ((*it_element).p[1] - globalcostmap.info.origin.position.y) / globalcostmap.info.resolution ) ;

			int gmidx = globalcostmap.info.width * ncmy	+ ncmx ;
			bool bisexplored = cmdata[gmidx] >=0 ? true : false ;

			//bool bisexplored = is_explored(ncmx, ncmy, globalcostmap.info.width, cmdata) ;

			if( bisexplored )
			{
				ROS_DEBUG("removing ufpt (%f %f) from the unreachable fpt list \n", (*it_element).p[0], (*it_element).p[1]) ;
				m_unreachable_frontier_set.erase(it_element);
			}
		}
	}

}

int FrontierDetectorDMS::selectNextBestPoint( const geometry_msgs::PoseStamped& robotpose, const nav_msgs::Path& goalexclusivefpts, geometry_msgs::PoseStamped& nextbestpoint  )
{
	std::vector<cv::Point2f> cvfrontierpoints;
	cv::Point2f cvrobotpoint( robotpose.pose.position.x, robotpose.pose.position.y );

	for (const auto & pi : goalexclusivefpts.poses)
		cvfrontierpoints.push_back(  cv::Point2f(pi.pose.position.x, pi.pose.position.y) );

	std::sort( begin(cvfrontierpoints), end(cvfrontierpoints), [cvrobotpoint](const cv::Point2f& lhs, const cv::Point2f& rhs)
			{ return FrontierDetectorDMS::euc_dist(cvrobotpoint, lhs) < FrontierDetectorDMS::euc_dist(cvrobotpoint, rhs); });

	for (const auto & pi : cvfrontierpoints )
	{
		float fdist = euc_dist( pi, cvrobotpoint ) ;
		//ROS_INFO("dist to alternative goals (%f %f) : %f \n", pi.x, pi.y, fdist );
	}

//	ROS_ASSERT( ++mn_prev_nbv_posidx >= 0 ); // to prev possible oscillation in selecting next best point

	ROS_WARN("The next best target is < %f %f > \n", cvfrontierpoints[0].x, cvfrontierpoints[0].y);

	nextbestpoint.pose.position.x = cvfrontierpoints[0].x ;
	nextbestpoint.pose.position.y = cvfrontierpoints[0].y ;
	nextbestpoint.header.frame_id = m_worldFrameId ;
	nextbestpoint.header.stamp = ros::Time::now();

	return 1;
}


int FrontierDetectorDMS::selectEscapingPoint( geometry_msgs::PoseStamped& escapepoint)
{
	//
}

int FrontierDetectorDMS::moveBackWard()
{
	ROS_INFO("+++++++++++++++++ moving backward  ++++++++++++++\n");
	geometry_msgs::Twist cmd_vel;
	cmd_vel.linear.x = -0.5;
	cmd_vel.linear.y = 0.0;
	cmd_vel.angular.z = 0.0;

	uint32_t start_time = ros::Time::now().sec ;
	uint32_t curr_time = start_time ;
	while( curr_time < start_time + 2 )
	{
		m_velPub.publish(cmd_vel);
		curr_time = ros::Time::now().sec ;
	}
	cmd_vel.angular.x = 0.0;
	m_velPub.publish(cmd_vel);
	ROS_INFO("+++++++++++++++++ end of moving backward +++++++++\n");
}



// mapcallback for dynamic mapsize (i.e for the cartographer)
void FrontierDetectorDMS::mapdataCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) //const octomap_server::mapframedata& msg )
{

ROS_INFO("********** \t start mapdata callback routine \t ********** \n");

ros::WallTime	mapCallStartTime = ros::WallTime::now();
	//ROS_INFO("@ mapdataCallback() ");

	if(!mb_isinitmotion_completed)
	{
		ROS_WARN("FD has not fully instantiated yet !");
		return;
	}

	if(m_robotvel.linear.x == 0 && m_robotvel.angular.z == 0 ) // robot is physically stopped
	{
		const std::unique_lock<mutex> lock(mutex_robot_state) ;
		me_robotstate = ROBOT_STATE::ROBOT_IS_NOT_MOVING;
	}

//	nav_msgs::OccupancyGrid globalcostmap;
	m_globalcostmap = *msg ;
	const nav_msgs::OccupancyGrid globalcostmap = m_globalcostmap;

ROS_INFO("got costmap size of %d %d \n", m_globalcostmap.info.height, m_globalcostmap.info.width );

	generateGridmapFromCostmap();

	float cmresolution=globalcostmap.info.resolution;
	float gmresolution= cmresolution ;
	float cmstartx=globalcostmap.info.origin.position.x;
	float cmstarty=globalcostmap.info.origin.position.y;
	uint32_t cmwidth =globalcostmap.info.width;
	uint32_t cmheight=globalcostmap.info.height;
	std::vector<signed char> cmdata, gmdata;

	cmdata =globalcostmap.data;
	gmdata = m_gridmap.data;

	// set the cent of the map as the init robot position (x, y)
	// cv::Point Offset = compute_rpose_wrt_maporig() ;
	geometry_msgs::PoseStamped start_world = GetCurrRobotPose( );
	cv::Point Offset = world2gridmap( cv::Point2f( 0.f, 0.f ) ) ; // (0, 0) loc w.r.t the gmap orig (left, top)

	mn_roi_origx = mn_globalmap_centx - Offset.x; // ox in gmap coordinate when the cent of gmap is the map orig
	mn_roi_origy = mn_globalmap_centy - Offset.y; // ox in gmap coordinate when the cent of gmap is the map orig
	cv::Rect roi_active( mn_roi_origx, mn_roi_origy, mn_activemap_width, mn_activemap_height); //cmwidth, cmheight );
	cv::Point start_gm = world2gridmap( cv::Point2f( start_world.pose.position.x, start_world.pose.position.y ) );

ROS_INFO("roi: %d %d %d %d \n", mn_roi_origx, mn_roi_origy, cmwidth, cmheight );
ROS_INFO("ox oy: %f %f rx_w ry_w: %f %f rx_g ry_g: %d %d\n", globalcostmap.info.origin.position.x, globalcostmap.info.origin.position.y,
		start_world.pose.position.x, start_world.pose.position.y, start_gm.x, start_gm.y ) ;

ROS_ASSERT(0);
// set/init global gridmap and costmap images (mcvu_globalmapimg, mcvu_globalcostmap)
	initGlobalmapimgs( cmheight, cmwidth, globalcostmap );

// The robot is not moving (or ready to move)... we can go ahead plan the next action...
// i.e.) We locate frontier points again, followed by publishing the new goal

	cv::Mat map_active ; // partial (on reconstructing map)
	map_active = mcvu_globalmapimg( roi_active );

/*******************************************************************************************************************************************
 We need to zero-pad around img b/c m_gridmap dynamically increases
 u = unk padding (offset), x = orig img contents
 Note that this padding has nothing to do with the extra canvas size allocated in mcvu_globalmapimg... The reason for large mcvu_globalmapimg is to standardize the full map size (perhaps for DL tranning)
 while ROI_OFFSET ensures FFP search begins from an UNKNOWN cell. That is FFP scans on the ROI (including the ROI_OFFSET region) of mcvu_globalmapimg
*********************************************************************************************************************************************/

// u u u u u u u u
// u x x x x x x u
// u x x x x x x u
// u x x x x x x u
// u u u u u u u u

ROS_INFO("roi info: <%d %d %d %d> \n", roi_active.x, roi_active.y, roi_active.width, roi_active.height) ;

//  ROI_OFFSET is needed for FFP seed. Not for neuro-explorer. So, we disable them ...
//	uint8_t ukn = static_cast<uchar>(dffp::MapStatus::UNKNOWN) ;
//	cv::Mat img_plus_offset = cv::Mat( img_active.rows + ROI_OFFSET*2, img_active.cols + ROI_OFFSET*2, CV_8U, cv::Scalar(ukn) ) ;
//	cv::Rect myroi( ROI_OFFSET, ROI_OFFSET, img_active.cols, img_active.rows );
//	cv::Mat img_roi = img_plus_offset(myroi) ;
//	img_active.copyTo(img_roi) ;

	cv::Mat processed_gmap = map_active.clone() ;
	if( mn_numpyrdownsample > 0)
	{
		// be careful here... using pyrDown() interpolates occ and free, making the boarder area (0 and 127) to be 127/2 !!
		// 127 reprents an occupied cell !!!
		for(int iter=0; iter < mn_numpyrdownsample; iter++ )
		{
			int nrows = processed_gmap.rows; //% 2 == 0 ? img.rows : img.rows + 1 ;
			int ncols = processed_gmap.cols; // % 2 == 0 ? img.cols : img.cols + 1 ;
			//ROS_INFO("sizes orig: %d %d ds: %d %d \n", img_.rows, img_.cols, nrows/2, ncols/2 );
			pyrDown(processed_gmap, processed_gmap, cv::Size( ncols/2, nrows/2 ) );
			clusterToThreeLabels( processed_gmap );
		}
	}


//cv::imwrite("/home/hankm/results/neuro_exploration_res/gmap_pad.png", mcvu_globalmapimg);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/processed_gmap.png", processed_gmap);

// DNN FFP detection
	cv::Mat fr_img = processed_gmap.clone();
ROS_INFO("begin DNN FR detection processed gmap size: %d %d \n", processed_gmap.rows, processed_gmap.cols);

//////////////////////////////////////////////////////////////////////////////////////////////
// 						run FD prediction session
//////////////////////////////////////////////////////////////////////////////////////////////
ros::WallTime GFFPstartTime = ros::WallTime::now();
	run_tf_fr_detector_session(processed_gmap, fr_img);
ros::WallTime GFFPendTime = ros::WallTime::now();
double ffp_time = (GFFPendTime - GFFPstartTime ).toNSec() * 1e-6;
ROS_INFO("done FR prediction \n");

//cv::imwrite("/home/hankm/results/neuro_exploration_res/fr_img.png", fr_img);

// get robot pose in the shifted gm image coordinate
	cv::Point rpos_gm = world2gridmap( cv::Point2f( start_world.pose.position.x, start_world.pose.position.y ) ) ; 	// rpose in orig gm
	//rpos_gm = cv::Point( (rpos_gm.x + roi.x) / mn_scale, (rpos_gm.y + roi.y) / mn_scale ) ;  			// rpos in padded img --> rpos in ds img
	cv::Point rpos_gmds = cv::Point( rpos_gm.x / mn_scale, rpos_gm.y / mn_scale ) ;  // rpose in active map down sampled (512 x 512 )

// cp fr_img (local) to global fr_img
	copyFRtoGlobalmapimg( roi_active, fr_img );
//cv::imwrite("/home/hankm/results/neuro_exploration_res/global_frimg_ds.png", mcvu_globalfrimg_ds);
ROS_INFO("done copying fr_img to mcvu_globalfrimg_ds \n");

	cv::Mat gmap_tform, fr_img_tformed, astar_net_input_tformed ;
	m_data_handler.transform_map_to_robotposition(fr_img, rpos_gmds.x, rpos_gmds.y, 0, fr_img_tformed) ;  // tform fr_img
	m_data_handler.transform_map_to_robotposition(processed_gmap, rpos_gmds.x, rpos_gmds.y, 127, gmap_tform) ;  // tform fr_img
	cv::Mat gaussimg_32f = m_data_handler.GetGaussianImg();
	m_data_handler.generate_astar_net_input(fr_img_tformed, gmap_tform, gaussimg_32f, astar_net_input_tformed);
//cv::Mat bgr[3] ;
//split(astar_net_input, bgr) ;
//cv::Mat I0,I1,I2, G ;
//bgr[0].convertTo(I0, CV_8U, 255.f) ;
//bgr[1].convertTo(I1, CV_8U, 255.f) ;
//bgr[2].convertTo(I2, CV_8U, 255.f) ;
//gaussimg_32f.convertTo(G, CV_8U, 255.f);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/img0.png", I0);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/img1.png", I1);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/img2.png", I2);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/gauss.png", G);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/gmap_tfrom.png", gmap_tform);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/gauss_img.png", gaussimg_32f * 255.f);

//cv::imwrite("/home/hankm/results/neuro_exploration_res/astar_net_input.png", astar_net_input_tformed * 255.f);

//////////////////////////////////////////////////////////////////////////////////////////////
// 						run astar prediction
//////////////////////////////////////////////////////////////////////////////////////////////

ros::WallTime GPstartTime = ros::WallTime::now();
	cv::Mat potmap_prediction_tformed ;
	run_tf_astar_session( astar_net_input_tformed, potmap_prediction_tformed) ;
ros::WallTime GPendTime = ros::WallTime::now();
double planning_time = (GPendTime - GPstartTime ).toNSec() * 1e-6;
ROS_INFO("done potmap prediction \n");
//cv::imwrite("/home/hankm/results/neuro_exploration_res/potmap_prediction.png", potmap_prediction);

ROS_INFO(" rpos_gm: %d %d  potmap size: %d %d \n", rpos_gm.x, rpos_gm.y, potmap_prediction_tformed.rows, potmap_prediction_tformed.cols );
	cv::Mat potmap_prediction_corrected;
	m_data_handler.inv_transform_map_to_robotposition(potmap_prediction_tformed, rpos_gmds.x, rpos_gmds.y, 0, potmap_prediction_corrected);
ROS_INFO("done potmap tform \n");
//cv::imwrite("/home/hankm/results/neuro_exploration_res/potmap_corrected.png", potmap_prediction_corrected );

// The last tform to original coordinate
//	int xoffset = roi.x / mn_scale ; // roi is defined in max_global_height (1024)
//	int yoffset = roi.y / mn_scale ;
//ROS_INFO("eliminating roi_offset <%d %d> \n", xoffset, yoffset);
//	for( int jj=0; jj < potmap_prediction_corrected.size(); jj++)
//	{
//		potmap_prediction_corrected[jj].x = potmap_prediction_corrected[jj].x - xoffset ;
//		potmap_prediction_corrected[jj].y = potmap_prediction_corrected[jj].y - yoffset ;
//	}

//////////////////////////////////////////////////////////////////////////////////////////////
// 						run covrew prediction
//////////////////////////////////////////////////////////////////////////////////////////////
ros::WallTime CRstartTime = ros::WallTime::now();
	cv::Mat covrew_prediction ;
	run_tf_covrew_session( processed_gmap, covrew_prediction ) ;
ros::WallTime CRendTime = ros::WallTime::now();
double covrew_time = (CRendTime - CRstartTime ).toNSec() * 1e-6;

//cv::imwrite("/home/hankm/results/neuro_exploration_res/covrew_net_output.png", covrew_prediction );
//ROS_INFO("done cov rew prediction \n");

//////////////////////////////////////////////////////////////////////////////////////////////
// 						Ensemble inv_potmap and covrew
//////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat ensembled_prediction = cv::Mat::zeros( potmap_prediction_corrected.rows, potmap_prediction_corrected.cols, potmap_prediction_corrected.depth() );
	ensemble_predictions(potmap_prediction_corrected, covrew_prediction, ensembled_prediction);

//cv::imwrite("/home/hankm/results/neuro_exploration_res/ensembled_prediction.png", ensembled_prediction );

	double covrew_minVal, covrew_maxVal;
	cv::minMaxLoc( covrew_prediction, &covrew_minVal, &covrew_maxVal );
	int ncovrew_maxVal = static_cast<int>(covrew_maxVal);
	ROS_INFO("max of covrew %f \n", covrew_maxVal);

	int xoffset_ds = roi_active.x / mn_scale ; // roi is defined in max_global_height (2048)
	int yoffset_ds = roi_active.y / mn_scale ;

//	vector<PointClass> covrew_labeled_points; //labeled_points_corrected ;
//	int num_labeled_points = assign_classes_to_points( covrew_prediction, covrew_labeled_points);
//	PointClassSet covrew_point_class( rgb(0.f, 0.f, 1.f), rgb(1.f, 1.f, 0.f), ncovrew_maxVal ) ;
//    vector<cv::Point> covrew_optpt_cands ;
//	for(int ii=0; ii < covrew_labeled_points.size(); ii++)
//	{
//		covrew_point_class.push_point( PointClass( covrew_labeled_points[ii].x, covrew_labeled_points[ii].y, covrew_labeled_points[ii].label ) ) ;
//		if(covrew_labeled_points[ii].label == ncovrew_maxVal )
//		{
//			covrew_optpt_cands.push_back( cv::Point( covrew_labeled_points[ii].x, covrew_labeled_points[ii].y) );
//		}
//	}

//	double potmap_minVal, potmap_maxVal;
//	cv::minMaxLoc( potmap_prediction_corrected, &potmap_minVal, &potmap_maxVal );
//	int npotmap_maxVal = static_cast<int>(potmap_maxVal);
//	vector<PointClass> potmap_labeled_points; //labeled_points_corrected ;
//	assign_classes_to_points( potmap_prediction_corrected, potmap_labeled_points);
//	PointClassSet potmap_point_class( rgb(0.f, 0.f, 1.f), rgb(1.f, 1.f, 0.f), npotmap_maxVal ) ;
//    vector<cv::Point> potmap_optpt_cands ;
//	for(int ii=0; ii < potmap_labeled_points.size(); ii++)
//	{
//		potmap_point_class.push_point( PointClass( potmap_labeled_points[ii].x, potmap_labeled_points[ii].y, potmap_labeled_points[ii].label ) ) ;
//		if( potmap_labeled_points[ii].label == npotmap_maxVal )
//		{
//			potmap_optpt_cands.push_back( cv::Point( potmap_labeled_points[ii].x, potmap_labeled_points[ii].y) );
//		}
//	}

	double ensembled_minVal, ensembled_maxVal;
	cv::minMaxLoc( ensembled_prediction, &ensembled_minVal, &ensembled_maxVal );
	int nensembled_maxVal = static_cast<int>(ensembled_maxVal);
	vector<PointClass> ensembled_labeled_points; //labeled_points_corrected ;
	int num_labeled_points = assign_classes_to_points( ensembled_prediction, ensembled_labeled_points);
	//std::sort(ensembled_labeled_points.begin(), ensembled_labeled_points.end(), class_cmp);  // causing double free corruption if used as is...
	PointClassSet ensembled_point_class( rgb(0.f, 0.f, 1.f), rgb(1.f, 1.f, 0.f), nensembled_maxVal ) ;

//    vector<cv::Point> ensembled_optpt_cands ;
//	for(int ii=0; ii < ensembled_labeled_points.size(); ii++)
//	{
//ROS_INFO("%d %d %d %d", ii, ensembled_labeled_points[ii].x, ensembled_labeled_points[ii].y, ensembled_labeled_points[ii].label );
//		ensembled_point_class.push_point( PointClass( ensembled_labeled_points[ii].x , ensembled_labeled_points[ii].y , ensembled_labeled_points[ii].label ) ) ;
//		if( ensembled_labeled_points[ii].label == nensembled_maxVal )
//		{
//			ensembled_optpt_cands.push_back( cv::Point( ensembled_labeled_points[ii].x , ensembled_labeled_points[ii].y ) );
//		}
//	}
//	vector<cv::Point> optpt_cands = ensembled_optpt_cands ;

//ROS_INFO(" num labeled points/ ensembed opt points: %d %d\n", num_labeled_points, ensembled_optpt_cands.size() );
//ROS_INFO("img_frontiers_offset (DNN output) size: %d %d \n", fr_img.rows, fr_img.cols );
// locate the most closest labeled points w.r.t the centroid pts

//	vector<vector<cv::Point> > global_contours_plus_offset;
//	vector<cv::Vec4i> hierarchy;
//	cv::findContours( mcvu_globalfrimg_ds, global_contours_plus_offset, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

	cv::Mat cvopt_ensembled_prediction ;
	cv::threshold(ensembled_prediction, cvopt_ensembled_prediction, ensembled_maxVal-1, 255, cv::THRESH_BINARY);
//	vector<vector<cv::Point> > opt_contours_plus_offset ;
//	vector<cv::Vec4i> opt_hierarchy;
//	cv::findContours( cvopt_ensembled_prediction, opt_contours_plus_offset, opt_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

//cv::imwrite("/home/hankm/results/neuro_exploration_res/opt_ensembled_prediction.png", cvopt_ensembled_prediction);

#ifdef FD_DEBUG_MODE
	string outfilename =  m_str_debugpath + "/global_mapimg.png" ;
//	cv::imwrite( outfilename.c_str(), m_uMapImg);
//	cv::imwrite(m_str_debugpath + "/labeled_img.png", img);
//	cv::imwrite(m_str_debugpath + "/img_frontiers.png",img_frontiers);
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////// 		update unreachable frontiers;
//////////////////////////////////////////////////////////////////////////////////////////////
	updateUnreachablePointSet( globalcostmap );

	// init curr frontier point sets
	{
		const std::unique_lock<mutex> lock_prev(mutex_prev_frontier_set);
		const std::unique_lock<mutex> lock_curr(mutex_curr_frontier_set);
		const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set);
		// append valid previous frontier points after the sanity check
		for (const auto & pi : m_prev_frontier_set)
		{
//ROS_INFO("checking reachability %f %f \n", pi.p[0], pi.p[1] );
			int ngmx = static_cast<int>( (pi.p[0] - globalcostmap.info.origin.position.x) / cmresolution ) ;
			int ngmy = static_cast<int>( (pi.p[1] - globalcostmap.info.origin.position.y) / cmresolution ) ;
			if( frontier_sanity_check(ngmx, ngmy, cmwidth, cmdata) ) //if( cmdata[ ngmy * cmwidth + ngmx ] == -1 )
			{
				if( mo_frontierfilter.isReachable( m_unreachable_frontier_set, pi.p[0], pi.p[1] ) )
				{
					pointset pnew( pi.p[0], pi.p[1] );
					m_curr_frontier_set.insert( pnew );
				}
			}
		}
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Note that model_output_contours are found in DNN output img in which contains extra padding ( using roi ) to make fixed size: 512 512 img.
// THe actual contours <contours_plus_offset> must be the ones found in actual gridmap (dynamic sized), thus we must remove the roi offset !!
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	vector<FrontierPoint> voGlobalFpts ;
	int num_global_frontier_point = locateFptsFromFRimg(mcvu_globalfrimg_ds, 0, 0, voGlobalFpts) ;

	vector<FrontierPoint> voLocalOptFpts ;
	int num_opt_frontier_point = locateFptsFromFRimg(cvopt_ensembled_prediction, 0, 0, voLocalOptFpts) ;

ROS_INFO("global / local fpts : %d %d \n", num_global_frontier_point, num_opt_frontier_point) ;

cv::imwrite("/home/hankm/results/neuro_exploration_res/globalfrimg_ds.png", mcvu_globalfrimg_ds);
cv::imwrite("/home/hankm/results/neuro_exploration_res/opt_ensembled_prediction.png", cvopt_ensembled_prediction);

ROS_ASSERT(0);

	// set viz markers
	// init fr viz markers
	visualization_msgs::Marker vizfrontier_regions;
	vizfrontier_regions = SetVizMarker( 0, visualization_msgs::Marker::ADD, 0.f, 0.f, 0.f, m_worldFrameId, 1.f, 0.f, 0.f, 0.1 );
	vizfrontier_regions.type = visualization_msgs::Marker::POINTS;

//	for( int ii=0; ii < global_fr.size(); ii++ )
//	{
//		for( int jj=0; jj < global_fr[ii].size(); jj++)
//		{
//			global_fr[ii][jj].x = global_contours_plus_offset[ii][jj].x  - xoffset_ds ;
//			global_fr[ii][jj].y = global_contours_plus_offset[ii][jj].y  - yoffset_ds ;
//			FrontierPoint oFRpoint( cv::Point(global_fr[ii][jj].x - ROI_OFFSET, global_fr[ii][jj].y - ROI_OFFSET), cmheight, cmwidth,
//									m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x, gmresolution, mn_numpyrdownsample );
//			geometry_msgs::Point point_w;
//			point_w.x = oFRpoint.GetInitWorldPosition().x ;
//			point_w.y = oFRpoint.GetInitWorldPosition().y ;
//			vizfrontier_regions.points.push_back( point_w ) ;
////ROS_INFO("global FR region pts  gm: <%d %d>  world <%f %f>\n", contours_plus_offset[ii][jj].x, contours_plus_offset[ii][jj].y, point_w.x, point_w.y );
//		}
//	}

	for( int ii=0; ii < voGlobalFpts.size(); ii++ )
	{
		FrontierPoint oPoint = voGlobalFpts[ii];
		if( oPoint.isConfidentFrontierPoint() )
		{
			int nx_corrected = oPoint.GetCorrectedGridmapPosition().x - xoffset_ds ;
			int ny_corrected = oPoint.GetCorrectedGridmapPosition().y - yoffset_ds ;
			FrontierPoint oFRpoint( cv::Point(nx_corrected, ny_corrected ), cmheight, cmwidth,
									m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x, gmresolution, mn_numpyrdownsample );
			geometry_msgs::Point point_w;
			point_w.x = oFRpoint.GetInitWorldPosition().x ;
			point_w.y = oFRpoint.GetInitWorldPosition().y ;
			vizfrontier_regions.points.push_back( point_w ) ;
		}
//ROS_INFO("global FR region pts  gm: <%d %d>  world <%f %f>\n", contours_plus_offset[ii][jj].x, contours_plus_offset[ii][jj].y, point_w.x, point_w.y );
	}


ROS_INFO("global FR region pts size: %d \n", vizfrontier_regions.points.size() );

	visualization_msgs::Marker vizoptastar_regions;
	visualization_msgs::Marker vizoptcov_regions; // max region of optcov map
	visualization_msgs::Marker vizoptensembled_regions; // max region of ensembled map
	visualization_msgs::Marker vizbound_lines ; // active bound lines

	// set cov opt viz marker
	//setVizMarkerFromPointClass( covrew_point_class, vizoptcov_regions, rgb(1.f, 0.f, 0.f), 0.1  );

	// set astar opt viz marker
	//setVizMarkerFromPointClass( potmap_point_class, vizoptastar_regions, rgb(1.f, 0.f, 0.f), 0.1 );

	// set ensembled viz marker
	setVizMarkerFromPointClass( ensembled_point_class, vizoptensembled_regions, rgb(1.f, 0.f, 0.f), 0.1 );

	// set active bound
	setActiveBound( start_world.pose.position.x, start_world.pose.position.y, mn_activemap_width, mn_activemap_height, vizbound_lines );
//
//	// optimal cands
//	vector<FrontierPoint> voFrontierCands;
//	int num_opt_frontier_point = 0;
//	if( opt_contours_plus_offset.size() == 0 )
//	{
//		ROS_WARN("There is no opt FR region to go from DNN process\n");
////		if( ensembled_labeled_points.size() == 0 )
////		{
////			ROS_WARN("There is no FR in the current input map \n");
////		}
////		else
////		{
////
////		}
//	}
//	else // We found opt fr in the current map. We append the new opt points to m_curr_frontier_set accordingly
//	{
////		FrontierPoint oFRpoint( cv::Point(opt_contours_plus_offset[0][0].x, opt_contours_plus_offset[0][0].y), cmheight, cmwidth,
////								m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x,
////							   cmresolution, mn_numpyrdownsample );
//
//		geometry_msgs::Point point_w;
//		vector<cv::Point> vecents_offset;  // shifted by the offet param
//		for(int i=0; i < opt_contours_plus_offset.size(); i++)
//		{
//			int nx =0, ny =0 ;
//			int ncnt = 0 ;
//			vector<cv::Point> opt_fr_plus_offset = opt_contours_plus_offset[i];
//			for( int j=0; j < opt_fr_plus_offset.size(); j++)
//			{
//				FrontierPoint oFRpoint( cv::Point(opt_fr_plus_offset[j].x - ROI_OFFSET, opt_fr_plus_offset[j].y - ROI_OFFSET), cmheight, cmwidth,
//										m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x,
//									   cmresolution, mn_numpyrdownsample );
//				point_w.x = oFRpoint.GetInitWorldPosition().x ;
//				point_w.y = oFRpoint.GetInitWorldPosition().y ;
//				vizfrontier_regions.points.push_back( point_w ) ;
//				nx += opt_fr_plus_offset[j].x ;
//				ny += opt_fr_plus_offset[j].y ;
//	//			ROS_INFO("%d %d / %d %d \n", contour_plus_offset[j].x, contour_plus_offset[j].y, img_plus_offset.rows, img_plus_offset.cols );
//				ncnt += 1;
//			}
//			int ncx = nx / ncnt ;
//			int ncy = ny / ncnt ;
//
//			cv::Point ncent( ncx,  ncy ) ;
//			vecents_offset.push_back( ncent );
//		}
//		CV_Assert( opt_contours_plus_offset.size() == vecents_offset.size() );
//
//		for( int i = 0; i < opt_contours_plus_offset.size(); i++ )
//		{
//			vector<cv::Point> opt_fr_plus_offset = opt_contours_plus_offset[i] ;
//			if(opt_fr_plus_offset.size() < m_frontiers_region_thr ) // don't care about small frontier regions
//				continue ;
//
//			int ncentx_offset = vecents_offset[i].x ;
//			int ncenty_offset = vecents_offset[i].y ;
//
//			int nmindist = 10000000 ;
//			int nmindistidx = -1;
//
//			for (int j=0; j < opt_fr_plus_offset.size(); j++)
//			{
//				int nx = opt_fr_plus_offset[j].x ;
//				int ny = opt_fr_plus_offset[j].y ;
//				int ndist = std::sqrt( (nx - ncentx_offset) * (nx - ncentx_offset) + (ny - ncenty_offset) * (ny - ncenty_offset) );
//				if(ndist < nmindist)
//				{
//					nmindist = ndist ;
//					nmindistidx = j ;
//				}
//			}
//
//			CV_Assert(nmindistidx >= 0);
//			cv::Point frontier_offset = opt_fr_plus_offset[nmindistidx];
//
//			//ROS_WARN(" %d %d \n", frontier_offset.x, frontier_offset.y);
//
//			cv::Point frontier ;
//			frontier.x = frontier_offset.x - ROI_OFFSET ;
//			frontier.y = frontier_offset.y - ROI_OFFSET ;
//
//			FrontierPoint oPoint( frontier, cmheight, cmwidth,
//									m_gridmap.info.origin.position.y, m_gridmap.info.origin.position.x,
//						   cmresolution, mn_numpyrdownsample );
//
//	/////////////////////////////////////////////////////////////////////
//	// 				We need to run position correction here
//	/////////////////////////////////////////////////////////////////////
//			cv::Point init_pt 		= oPoint.GetInitGridmapPosition() ; 	// position @ ds0 (original sized map)
//			cv::Point corrected_pt	= oPoint.GetCorrectedGridmapPosition() ;
//			correctFrontierPosition( m_gridmap, init_pt, mn_correctionwindow_width, corrected_pt  );
//
//			oPoint.SetCorrectedCoordinate(corrected_pt);
//			voFrontierCands.push_back(oPoint);
//		}
//
//		// run filter
//		const float fcm_conf_thr = mo_frontierfilter.GetCostmapConf() ;
//		const float fgm_conf_thr = mo_frontierfilter.GetGridmapConf() ;
//
//		// eliminate frontier points at obtacles
//
//		if( globalcostmap.info.width > 0 )
//		{
//			mo_frontierfilter.measureCostmapConfidence(globalcostmap, voFrontierCands);
////ROS_INFO("done CM filter \n");
//			mo_frontierfilter.measureGridmapConfidence(m_gridmap, voFrontierCands);
////ROS_INFO("done GM filter \n");
//
//			for(size_t idx=0; idx < voFrontierCands.size(); idx++)
//			{
//				cv::Point frontier_in_gm = voFrontierCands[idx].GetCorrectedGridmapPosition();
//				bool bisfrontier = is_frontier_point(frontier_in_gm.x, frontier_in_gm.y, cmwidth, cmheight, gmdata );
//				int gmidx = cmwidth * frontier_in_gm.y	+	frontier_in_gm.x ;
//				bool bisexplored = cmdata[gmidx] >=0 ? true : false ;
//				voFrontierCands[idx].SetFrontierFlag( fcm_conf_thr, fgm_conf_thr, bisexplored, bisfrontier );
//
////ROS_INFO("%f %f %d %d\n", voFrontierCands[idx].GetInitWorldPosition().x, voFrontierCands[idx].GetInitWorldPosition().y,
////						  voFrontierCands[idx].GetCorrectedGridmapPosition().x, voFrontierCands[idx].GetCorrectedGridmapPosition().y);
//
//			}
//			set<pointset> unreachable_frontiers;
//			{
//				const std::unique_lock<mutex> lock_unrc(mutex_unreachable_frontier_set) ;
//				unreachable_frontiers = m_unreachable_frontier_set ;
//			}
//			mo_frontierfilter.computeReachability( unreachable_frontiers, voFrontierCands );
//		}
//
//		{
//			const std::unique_lock<mutex> lock(mutex_curr_frontier_set);
//			for (size_t idx=0; idx < voFrontierCands.size(); idx++)
//			{
////ROS_INFO("<%f %f> confidence  %d \n", voFrontierCands[idx].GetCorrectedWorldPosition().x, voFrontierCands[idx].GetCorrectedWorldPosition().y,
////		voFrontierCands[idx].isConfidentFrontierPoint() );
//				if( voFrontierCands[idx].isConfidentFrontierPoint() )
//				{
//					//valid_frontier_indexs.push_back( idx );
//					cv::Point2f frontier_in_world = voFrontierCands[idx].GetCorrectedWorldPosition();
//					pointset pt( frontier_in_world.x, frontier_in_world.y );
//					m_curr_frontier_set.insert( pt );
//					num_opt_frontier_point++;
//				}
//			}
//		}
//	}


ROS_INFO(" DNN found : %d confident points while tot fpts is : %d \n", num_opt_frontier_point, m_curr_frontier_set.size() );
	//frontier_summary( voFrontierCands );

	publishFrontierRegionMarkers ( vizfrontier_regions );
	//publishOptCovRegionMarkers ( vizoptcov_regions );
	//publishOptAstarRegionMarkers( vizoptastar_regions );
	publishOptEnsembledRegionMarkers( vizoptensembled_regions );
	publishFrontierPointMarkers( ) ;
	publishActiveBoundLines( vizbound_lines );

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 		generate a path trajectory
// 		call make plan service
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ROS_INFO(" FR is found let's start the make plan service \n");

///////////////////////////////////////////////////////////////////////
// 1. estimate dist to each goal using euclidean distance heuristic (we need sorting here)
///////////////////////////////////////////////////////////////////////
	float fstartx = static_cast<float>( start_world.pose.position.x ) ;
	float fstarty = static_cast<float>( start_world.pose.position.y ) ;

//////////////////////////////////////////////////////////////////////////////////
// 2. use the fp corresponds to the min distance as the init fp. epsilon = A*(fp)
// 	i)  We first sort fpts based on their euc heuristic(), then try makePlan() for each of fpts in turn.
// 	ii) We need to sort them b/c the one with best heuristic could fail
//////////////////////////////////////////////////////////////////////////////////

	geometry_msgs::PoseStamped goal;
	nav_msgs::Path msg_frontierpoints ;

	if(num_opt_frontier_point == 0) // nothing
	{
ROS_WARN("DNN process has found no interesting FR. Go over all available fpts in the global map \n");
		// get cands from global frontier sets
		for( const auto & pi : m_curr_frontier_set)
		{
			geometry_msgs::PoseStamped tmp_goal = StampedPosefromSE2( pi.p[0], pi.p[1], 0.f );
			tmp_goal.header.frame_id = m_worldFrameId ;

			double fdist_sq = (start_world.pose.position.x - tmp_goal.pose.position.x ) * (start_world.pose.position.x - tmp_goal.pose.position.x ) +
					( start_world.pose.position.y - tmp_goal.pose.position.y ) * ( start_world.pose.position.y - tmp_goal.pose.position.y ) ;
			float fdist = sqrtf( static_cast<float>(fdist_sq) );
//ROS_WARN("fdist from <%f %f> to <%f %f> is  %f  %f\n", start.pose.position.x, start.pose.position.y, tmp_goal.pose.position.x, tmp_goal.pose.position.y, fdist_sq, fdist);
			if (fdist > 1.f)
				msg_frontierpoints.poses.push_back(tmp_goal);
		}

		if( msg_frontierpoints.poses.size() == 0)
		{
ROS_WARN("DNN process has found no interesting FR. There is no opt fpts available in  curr_frontier_set. \n  We look for the FR regions in global map ");

		}

ROS_WARN(" msg_frontierpoint from the global FR set : %d \n", msg_frontierpoints.poses.size() );
	}
	else	// found something in the current map
	{
		for( int idx=0; idx < voLocalOptFpts.size(); idx++)
		{
			if( voLocalOptFpts[idx].isConfidentFrontierPoint() )
			{
				cv::Point2f frontier_in_world = voLocalOptFpts[idx].GetCorrectedWorldPosition();
				geometry_msgs::PoseStamped tmp_goal = StampedPosefromSE2( frontier_in_world.x, frontier_in_world.y, 0.f );
				tmp_goal.header.frame_id = m_worldFrameId ;

				double fdist_sq = (start_world.pose.position.x - tmp_goal.pose.position.x ) * (start_world.pose.position.x - tmp_goal.pose.position.x ) +
						( start_world.pose.position.y - tmp_goal.pose.position.y ) * ( start_world.pose.position.y - tmp_goal.pose.position.y ) ;
				float fdist = sqrtf( static_cast<float>(fdist_sq) );
	//ROS_INFO("fdist from <%f %f> to <%f %f> is  %f  %f \n", start.pose.position.x, start.pose.position.y, tmp_goal.pose.position.x, tmp_goal.pose.position.y, fdist_sq, fdist);
				if (fdist > 1.f)
					msg_frontierpoints.poses.push_back(tmp_goal);
			}
		}
ROS_INFO(" msg_frontierpoint found from DNN process size: %d  / %d \n", msg_frontierpoints.poses.size(), voLocalOptFpts.size() );
	}

	if( m_curr_frontier_set.empty() && msg_frontierpoints.poses.size() == 0 ) // terminating condition
	{
ROS_WARN("no more valid frontiers \n");
		// delete markers
		visualization_msgs::MarkerArray ftmarkers_old = m_frontierpoint_markers ;
		for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
			ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
		m_markerfrontierPub.publish(ftmarkers_old);
		ftmarkers_old = m_unreachable_markers ;
		for(size_t idx=0; idx < ftmarkers_old.markers.size(); idx++)
			ftmarkers_old.markers[idx].action = visualization_msgs::Marker::DELETE; //SetVizMarker( idx, visualization_msgs::Marker::DELETE, 0.f, 0.f, 0.5, "map", 0.f, 1.f, 0.f );
		m_marker_unreachpointPub.publish(ftmarkers_old);
		mb_explorationisdone = true;
		return;
	}

ROS_INFO(" got valid frontier points \n");

	// publish goalexclusive fpts
	int tmpcnt = 0;
	nav_msgs::Path goalexclusivefpts ;
	goalexclusivefpts.header.frame_id = m_worldFrameId;
	goalexclusivefpts.header.seq = m_curr_frontier_set.size() -1 ;
	goalexclusivefpts.header.stamp = ros::Time::now();
	for (const auto & pi : m_curr_frontier_set)
	{
		if (tmpcnt != 0)
		{
			geometry_msgs::PoseStamped fpt = StampedPosefromSE2( pi.p[0], pi.p[1], 0.f ) ;
			fpt.header.frame_id = m_worldFrameId ;
			fpt.header.stamp = goalexclusivefpts.header.stamp ;
			goalexclusivefpts.poses.push_back(fpt);
		}
		tmpcnt++;
	}

//ROS_INFO(" done here 1 \n");

//cv::Point2f cvbestgoal = cvgoalcands[best_idx] ;  // just for now... we need to fix it later
//	geometry_msgs::PoseStamped best_goal = msg_frontierpoints.poses[best_idx] ; //ps.p[0], ps.p[1], 0.f );

// if the best frontier point is the same as the previous frontier point, we need to set a different goal

	geometry_msgs::PoseStamped best_goal = msg_frontierpoints.poses[0] ; //ps.p[0], ps.p[1], 0.f );

	// check for ocsillation
	float fdist2prevposition = euc_dist( cv::Point2f( m_previous_robot_pose.pose.position.x, m_previous_robot_pose.pose.position.y ), cv::Point2f( start_world.pose.position.x, start_world.pose.position.y ) ) ;
	if( fdist2prevposition > 0.5 ) // 0.5 is ros nav stack default
	{
		m_last_oscillation_reset = ros::Time::now();
	}
//ROS_INFO(" done here 2 \n");

	bool bisocillating = (m_last_oscillation_reset + ros::Duration(8.0) < ros::Time::now() );
	if( bisocillating ) // first check for oscillation
	{
		ROS_WARN("Oscillation detected. Set <%f %f> as unreachable fpt \n", m_previous_goal.pose.pose.position.x, m_previous_goal.pose.pose.position.y);

//		publishGoalPointMarker( ) ;

		// publish current goal as unreachable pt
		geometry_msgs::PoseStamped ufpt = StampedPosefromSE2( m_previous_goal.pose.pose.position.x, m_previous_goal.pose.pose.position.y, 0.f ) ;
		appendUnreachablePoint(  ufpt ) ;
		publishUnreachableMarkers( );

		updatePrevFrontierPointsList( ) ;

		// update m_prev_frontier_set / m_curr_frontier_set list
		mb_strict_unreachable_decision = true;
		m_last_oscillation_reset = ros::Time::now();
		return ;
	}
	else if( equals_to_prevgoal( best_goal ) ) //|| me_prev_exploration_state == ABORTED ) // cond to select the next best alternative
	{
		ROS_WARN(" The best target <%f %f> found by the planner is the same as the previous goal. Need to select an alternative target. \n",
				best_goal.pose.position.x,  best_goal.pose.position.y );

		mb_nbv_selected = true ;
		if( goalexclusivefpts.poses.size() == 0 )
		{
			ROS_WARN(" However, there is no more frontier points to visit. \n");
			mb_explorationisdone = true;
			return;
		}

		geometry_msgs::PoseStamped ufpt = StampedPosefromSE2( best_goal.pose.position.x, best_goal.pose.position.y, 0.f ) ;
		appendUnreachablePoint(  ufpt ) ;

//		ROS_ASSERT( goalexclusivefpts.poses.size() >  mn_prev_nbv_posidx );

		// choose the next best goal based on the eucdist heurisitic.
ROS_WARN("The target goal is equal to the previous goal... Selecting NBV point from <%d goalexclusivefpts> to be the next best target \n", goalexclusivefpts.poses.size() );
		geometry_msgs::PoseStamped nextbestpoint = StampedPosefromSE2( 0.f, 0.f, 0.f ) ;
		selectNextBestPoint( start_world,  goalexclusivefpts, nextbestpoint) ;
ROS_WARN("Selecting the next best point since frontier pts is unreachable ..  \n");
		const std::unique_lock<mutex> lock(mutex_currgoal);
		m_targetgoal.header.frame_id = m_worldFrameId ;
		m_targetgoal.pose.pose = nextbestpoint.pose ;
		m_previous_goal = m_targetgoal ;
	}
	else // ordinary case ( choosing the optimal pt, then move toward there )
	{
		//mn_prev_nbv_posidx = -1;
		mb_nbv_selected = false ;
		const std::unique_lock<mutex> lock(mutex_currgoal);
		m_targetgoal.header.frame_id = m_worldFrameId ;
		m_targetgoal.pose.pose = best_goal.pose ;
		m_previous_goal = m_targetgoal ;
	}

//ROS_INFO(" done here 2 \n");

//////////////////////////////////////////////////////////////////////////////
// 						Publish frontier pts to Rviz						//
//////////////////////////////////////////////////////////////////////////////
	{
		const std::unique_lock<mutex> lock(mutex_robot_state) ;
		me_robotstate = ROBOT_STATE::ROBOT_IS_READY_TO_MOVE;
	}
//ROS_INFO(" done here 3 \n");
	updatePrevFrontierPointsList( ) ;
//ROS_INFO(" done here 4 \n");

////////////////////////////////////////////////////////////////////////////////////////////////////
// Lastly we publish the goal and other frontier points ( hands over the control to move_base )
////////////////////////////////////////////////////////////////////////////////////////////////////
	m_previous_robot_pose = start_world;
//	m_otherfrontierptsPub.publish(goalexclusivefpts);
	publishUnreachableMarkers( );
	m_currentgoalPub.publish(m_targetgoal);		// for control

ros::WallTime mapCallEndTime = ros::WallTime::now();
double mapcallback_time = (mapCallEndTime - mapCallStartTime).toNSec() * 1e-6;
ROS_INFO("\n "
		 " ********************************************************************************************* \n "
		 "     mapDataCallback exec time (ms): %f \n Potmap pred time (ms): %f  Cov est time (ms): %f \n "
		 " ********************************************************************************************* \n "
		, mapcallback_time, planning_time, covrew_time);

ROS_INFO("\n********** End of mapdata callback routine \t ********** \n");

	//saveDNNData( img_frontiers_offset, start, best_goal, best_plan, ROI_OFFSET, roi ) ;

// for timing
mf_totalcallbacktime_msec += mapcallback_time ;
mvf_fr_detection_time.push_back(ffp_time);
mvf_astar_time.push_back( planning_time ) ;
mvf_covrew_time.push_back(covrew_time) ;
mn_mapcallcnt++;

}

void FrontierDetectorDMS::run_tf_fr_detector_session( const cv::Mat& input_map, cv::Mat& model_output)
{
	int nheight = input_map.rows ;
	int nwidth  = input_map.cols ;
	memset( mpf_fd_data, 0.f, input_map.cols * input_map.rows*sizeof(float) );

    for(int ridx=0; ridx < input_map.rows; ridx++)
    {
    	for(int cidx=0; cidx < input_map.cols; cidx++)
    	{
    		int lidx = ridx * input_map.cols + cidx ;
    		mpf_fd_data[lidx] = static_cast<float>(input_map.data[lidx]) / 255.f   ;
    	}
    }
    const int ndata = sizeof(float)*1*input_map.rows*input_map.cols*1 ;
    const int64_t dims[] = {1, input_map.rows, input_map.cols, 1};
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, 4, mpf_fd_data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
    	//ROS_INFO("TF_NewTensor FR detection is OK\n");
    }
    else
    {
    	ROS_ERROR("ERROR: Failed creating FR detection TF_NewTensor\n");
    }

    mpptf_fd_input_values[0] = int_tensor;
    // //Run the Session

ros::WallTime SessionStartTime = ros::WallTime::now();
    TF_SessionRun(mptf_fd_Session, NULL, mptf_fd_input, mpptf_fd_input_values, 1, mptf_fd_output, mpptf_fd_output_values, 1, NULL, 0, NULL, mptf_fd_Status);
ros::WallTime SessionEndTime = ros::WallTime::now();
double session_time = (SessionEndTime - SessionStartTime).toNSec() * 1e-6;
mf_total_fd_sessiontime_msec = mf_total_fd_sessiontime_msec + session_time ;

    if(TF_GetCode(mptf_fd_Status) == TF_OK)
    {
    	//ROS_INFO("FR detection Session is OK\n");
    }
    else
    {
    	ROS_ERROR("%s",TF_Message(mptf_fd_Status));
    }
    cv::Mat res_32F(nheight, nwidth, CV_32FC1, TF_TensorData(*mpptf_fd_output_values));
	double minVal, maxVal;
	cv::minMaxLoc( res_32F, &minVal, &maxVal );
	model_output = ( res_32F > maxVal * 0.3 ) * 255 ;
}

void FrontierDetectorDMS::run_tf_astar_session( const cv::Mat& input_map, cv::Mat& model_output)
{
	int nheight = input_map.rows ;
	int nwidth  = input_map.cols ;
	int nchannels = input_map.channels() ;
	int nclasses = mn_num_classes ;

	memset( mpf_astar_data, 0.f, nheight * nwidth * nchannels * sizeof(float) );

	// C H W style { i.e.) depth first then shift to right }
	for(int ridx=0; ridx < nheight; ridx++)
	{
		for(int cidx=0; cidx < nwidth; cidx++)
		{
			//int lidx = ridx * 512 * 3 + cidx * NUM_CLASSES + chidx ;
			//int lidx = ridx * 512 + cidx + 512 * 512 * chidx ;
			float c0 = input_map.at<cv::Vec3f>(ridx, cidx)[0]   ;
			float c1 = input_map.at<cv::Vec3f>(ridx, cidx)[1]   ;
			float c2 = input_map.at<cv::Vec3f>(ridx, cidx)[2]   ;

			int idx0 = ridx * nheight * nchannels + cidx * nchannels + 0  ;
			int idx1 = ridx * nheight * nchannels + cidx * nchannels + 1  ;
			int idx2 = ridx * nheight * nchannels + cidx * nchannels + 2 ;
			mpf_astar_data[idx0] = c0 ;
			mpf_astar_data[idx1] = c1 ;
			mpf_astar_data[idx2] = c2 ;
		}
	}

    const int ndata = sizeof(float)*1*input_map.rows * input_map.cols * input_map.channels() ;
    const int64_t dims[] = {1, input_map.rows, input_map.cols, input_map.channels()};
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, 4, mpf_astar_data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
    	//ROS_INFO("A* TF_NewTensor is OK\n");
    }
    else
    {
    	ROS_ERROR("ERROR: Failed creating A* TF_NewTensor\n");
    }

    mpptf_astar_input_values[0] = int_tensor;
    // //Run the Session
ros::WallTime SessionStartTime = ros::WallTime::now();
    TF_SessionRun(mptf_astar_Session, NULL, mptf_astar_input, mpptf_astar_input_values, 1, mptf_astar_output, mpptf_astar_output_values, 1, NULL, 0, NULL, mptf_astar_Status);
ros::WallTime SessionEndTime = ros::WallTime::now();
double session_time = (SessionEndTime - SessionStartTime).toNSec() * 1e-6;
mf_total_astar_sessiontime_msec += session_time ;

    if(TF_GetCode(mptf_astar_Status) == TF_OK)
    {
    	//ROS_INFO("A* session is OK\n");
    }
    else
    {
    	//ROS_INFO("%s",TF_Message(mptf_astar_Status));
    }

	void* buff = TF_TensorData(mpptf_astar_output_values[0]);
	float* offsets = (float*)buff;

	model_output = cv::Mat::zeros(nheight, nwidth, CV_8U);

	// C H W style
	for (int ridx = 0; ridx < nheight; ++ridx)
	{
		for (int cidx = 0; cidx < nwidth; ++cidx)
		{
			int max_idx = -1;
			float max_val = -FLT_MAX;

			for(int chidx=0; chidx < nclasses; chidx++)	// hard coded for now...
			{
				int idx = ridx * nheight * nclasses + cidx * nclasses + chidx ;
				float val = offsets[idx] ;
				if (val > max_val)
				{
					max_val = val;
					max_idx = chidx;
				}
				//tmp.at<float>(ridx, cidx) = val ;
			}
			model_output.data[ridx*nheight + cidx] = max_idx ;
		}
	}
}

void FrontierDetectorDMS::run_tf_covrew_session( const cv::Mat& input_map, cv::Mat& model_output )
{
	int nheight = input_map.rows ;
	int nwidth  = input_map.cols ;
	int nclasses = mn_num_classes ;

	ROS_ASSERT(input_map.depth() == CV_8U);
	memset( mpf_covrew_data, 0.f, input_map.cols * input_map.rows*sizeof(float) );

    for(int ridx=0; ridx < input_map.rows; ridx++)
    {
    	for(int cidx=0; cidx < input_map.cols; cidx++)
    	{
    		int lidx = ridx * input_map.cols + cidx ;
    		mpf_covrew_data[lidx] = static_cast<float>(input_map.data[lidx]) / 255.f   ;
    	}
    }
    const int ndata = sizeof(float)*1*input_map.rows*input_map.cols*1 ;
    const int64_t dims[] = {1, input_map.rows, input_map.cols, 1};
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, 4, mpf_covrew_data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
    	//ROS_INFO("TF_NewTensor CovRew is OK\n");
    }
    else
    {
    	ROS_ERROR("ERROR: Failed creating CovRew TF_NewTensor\n");
    }
    mpptf_covrew_input_values[0] = int_tensor;
    // //Run the Session

ros::WallTime SessionStartTime = ros::WallTime::now();
    TF_SessionRun(mptf_covrew_Session, NULL, mptf_covrew_input, mpptf_covrew_input_values, 1, mptf_covrew_output, mpptf_covrew_output_values, 1, NULL, 0, NULL, mptf_covrew_Status);
ros::WallTime SessionEndTime = ros::WallTime::now();
double session_time = (SessionEndTime - SessionStartTime).toNSec() * 1e-6;
mf_total_covrew_sessiontime_msec = mf_total_covrew_sessiontime_msec + session_time ;

    if(TF_GetCode(mptf_covrew_Status) == TF_OK)
    {
    	//ROS_INFO("CovRew session is OK\n");
    }
    else
    {
    	ROS_ERROR("%s",TF_Message(mptf_covrew_Status));
    }

	void* buff = TF_TensorData(mpptf_covrew_output_values[0]);
	float* offsets = (float*)buff;

	model_output = cv::Mat::zeros(nheight, nwidth, CV_8U);

	// C H W style
	for (int ridx = 0; ridx < nheight; ++ridx)
	{
		for (int cidx = 0; cidx < nwidth; ++cidx)
		{
			int max_idx = -1;
			float max_val = -FLT_MAX;

			for(int chidx=0; chidx < nclasses; chidx++)	// hard coded for now...
			{
				int idx = ridx * nheight * nclasses + cidx * nclasses + chidx ;
				float val = offsets[idx] ;
				if (val > max_val)
				{
					max_val = val;
					max_idx = chidx;
				}
			}
			model_output.data[ridx*nheight + cidx] = max_idx ;
		}
	}
}

int FrontierDetectorDMS::locate_optimal_point_from_potmap( const cv::Mat& input_potmap, const uint8_t& optVal, vector<cv::Point>& points   )
{
	//cv::Mat optFR = input_potmap == optVal ;
	int nheight = input_potmap.rows ;
	int nwidth = input_potmap.cols ;
	int ncnt = 0;
	for( int ii = 0; ii < nheight; ii++ )
	{
		for(int jj =0; jj < nwidth; jj++)
		{
			int idx = nwidth * ii + jj ;
			if( input_potmap.data[ idx ] == optVal )
			{
				points.push_back(cv::Point(jj,ii));
				ncnt++;
			}
		}
	}
	return ncnt ;
}

int FrontierDetectorDMS::assign_classes_to_points( const cv::Mat& input_tmap, vector<PointClass>& points   )
{
	//cv::Mat optFR = input_potmap == optVal ;
	int nheight = input_tmap.rows ;
	int nwidth = input_tmap.cols ;
	int ncnt = 0;
	for( int ii = 0; ii < nheight; ii++ )
	{
		for(int jj =0; jj < nwidth; jj++)
		{
			int idx = nwidth * ii + jj ;
			uint8_t label = input_tmap.data[ idx ];
			if( label > 0 )
			{
				points.push_back( PointClass(jj, ii, (int)label) );
				ncnt++;
			}
		}
	}
	return ncnt ;
}

void FrontierDetectorDMS::ensemble_predictions( const cv::Mat& potmap_prediction, const cv::Mat& covrew_prediction, cv::Mat& ensembled_output )
{
	cv::addWeighted( potmap_prediction, 1.0, covrew_prediction, 0.0, 0.0, ensembled_output, -1 );
}

void FrontierDetectorDMS::saveDNNData( const cv::Mat& img_frontiers_offset, const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& best_goal,
								const std::vector<geometry_msgs::PoseStamped>& best_plan, const int& OFFSET, const cv::Rect& roi )
{
//	string strgmapimg 	= (boost::format("%s/mapimg%05d.png") % m_str_debugpath.c_str() % mn_mapcallcnt ).str() ;
//	string strmetadata 	= (boost::format("%s/metadata%05d.txt") % m_str_debugpath.c_str() % mn_mapcallcnt ).str() ;
//	string strimgFR		= (boost::format("%s/imgfr%05.txt") %  m_str_debugpath.c_str() % mn_mapcallcnt ).str() ;

	char cgmapimg[300], ccmimg[300], cmetadata[300], cimgFR[300], cbestplan[300];
	sprintf(cgmapimg,	"%s/gmimg%04d.png", m_str_debugpath.c_str(),mn_mapcallcnt);
	sprintf(ccmimg, 	"%s/cmimg%04d.png", m_str_debugpath.c_str(), mn_mapcallcnt );
	sprintf(cmetadata,	"%s/metadata%04d.txt", m_str_debugpath.c_str(),mn_mapcallcnt);
	sprintf(cimgFR,		"%s/frimg%04d.png", m_str_debugpath.c_str(),mn_mapcallcnt);
	sprintf(cbestplan,	"%s/bestplan%04d.txt", m_str_debugpath.c_str(), mn_mapcallcnt);

	// save gridmap and costmap
	unsigned char* pmap = mpo_costmap->getCharMap() ;
	cv::Mat cvcmapimg = cv::Mat( mcvu_globalmapimg.rows, mcvu_globalmapimg.cols, CV_8U);
	cv::imwrite( string(cgmapimg), mcvu_globalmapimg ) ;
	cv::imwrite( string(ccmimg), mcvu_costmapimg) ;
	// save metadata (robot pose, opt frontier point, )
	std::ofstream ofs_metadata( cmetadata );
	ofs_metadata <<
			start.pose.position.x << " " << start.pose.position.y << " " <<
			start.pose.orientation.x << " " <<	start.pose.orientation.y << " " << start.pose.orientation.z << " " << start.pose.orientation.w << " " <<
			best_goal.pose.position.x << " " << best_goal.pose.position.y << " " <<
			m_gridmap.info.height << " " << m_gridmap.info.width << " " << m_gridmap.info.origin.position.x << " " << m_gridmap.info.origin.position.y << " " << m_gridmap.info.resolution << " " <<
			OFFSET << " " << roi.x << " " << roi.y << " " << roi.height << " " << roi.width << endl;

	ofs_metadata.close();
	// save frontier region
	cv::imwrite( string(cimgFR), img_frontiers_offset );

	std::ofstream ofs_bestplan( cbestplan );
	for (int nidx=0; nidx < best_plan.size(); nidx++)
		ofs_bestplan << best_plan[nidx].pose.position.x << " " << best_plan[nidx].pose.position.y << endl;
	ofs_bestplan.close();
}


void FrontierDetectorDMS::doneCB( const actionlib::SimpleClientGoalState& state )
{
//    ROS_INFO("@DONECB: simpleClientGoalState [%s]", state.toString().c_str());

    if (m_move_client.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
         // do something as goal was reached
    	ROS_INFO("Touch down  \n");
		{
			const std::unique_lock<mutex> lock(mutex_robot_state) ;
			me_robotstate = ROBOT_STATE::ROBOT_IS_NOT_MOVING ;
		}
		me_prev_exploration_state = SUCCEEDED ;
    }
    else if (m_move_client.getState() == actionlib::SimpleClientGoalState::ABORTED)
    {
        // do something as goal was canceled
    	ROS_ERROR("MoveBase() has failed to reach the goal. The frontier point has aboarted ... \n");
		{
			const std::unique_lock<mutex> lock(mutex_robot_state) ;
			me_robotstate = ROBOT_STATE::ROBOT_IS_NOT_MOVING ;
		}
		me_prev_exploration_state = ABORTED ;
    }
    else
    {
    	ROS_ERROR("doneCB() received an unknown state %d \n", m_move_client.getState());
    	exit(-1);
    }
}


void FrontierDetectorDMS::moveRobotCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg )
{
// call actionlib
// robot is ready to move
	ROS_INFO("@moveRobotCallback Robot is < %s > \n ",  robot_state[me_robotstate+1] );

	if( me_robotstate >= ROBOT_STATE::FORCE_TO_STOP   )
		return;

	geometry_msgs::PoseWithCovarianceStamped goalpose = *msg ;
	{
		const std::unique_lock<mutex> lock(mutex_robot_state) ;
		me_robotstate = ROBOT_STATE::ROBOT_IS_MOVING ;
	}

	ROS_INFO("@moveRobotCallback received a plan\n");

	move_base_msgs::MoveBaseGoal goal;
	goal.target_pose.header.frame_id = m_worldFrameId; //m_baseFrameId ;
	goal.target_pose.header.stamp = ros::Time::now() ;

//	geometry_msgs::PoseWithCovarianceStamped goalpose = // m_pathplan.poses.back() ;

	goal.target_pose.pose.position.x = goalpose.pose.pose.position.x ;
	goal.target_pose.pose.position.y = goalpose.pose.pose.position.y ;
	goal.target_pose.pose.orientation.w = goalpose.pose.pose.orientation.w ;

	ROS_INFO("new destination target is set to <%f  %f> \n", goal.target_pose.pose.position.x, goal.target_pose.pose.position.y );

	publishUnreachableMarkers( ) ;

	// publish goal to Rviz
	if( !mb_nbv_selected )
	{
		publishGoalPointMarker( goalpose );
	}
	else
	{
//		m_targetgoal_marker.points.clear();
//		m_targetgoal_marker = SetVizMarker( -1, visualization_msgs::Marker::ADD, m_targetgoal.pose.pose.position.x, m_targetgoal.pose.pose.position.y, 0.f,
//				m_worldFrameId,	0.58f, 0.44f, 0.86f, (float)TARGET_MARKER_SIZE);
//		m_makergoalPub.publish(m_targetgoal_marker); // for viz
		publishGoalPointMarker( goalpose );
		mb_nbv_selected = false ;
	}

// inspect the path
//////////////////////////////////////////////////////////////////////////////////////////////
//ROS_INFO("+++++++++++++++++++++++++ @moveRobotCallback, sending a goal +++++++++++++++++++++++++++++++++++++\n");
ros::WallTime moveRobotStartTime = ros::WallTime::now();
	m_move_client.sendGoal(goal, boost::bind(&FrontierDetectorDMS::doneCB, this, _1), SimpleMoveBaseClient::SimpleActiveCallback() ) ;
//ROS_INFO("+++++++++++++++++++++++++ @moveRobotCallback, a goal is sent +++++++++++++++++++++++++++++++++++++\n");
	m_move_client.waitForResult();

ros::WallTime moveRobotEndTime = ros::WallTime::now();
double moverobot_time = (moveRobotEndTime - moveRobotStartTime).toNSec() * 1e-6;
mf_totalmotiontime_msec = mf_totalmotiontime_msec + moverobot_time ;
mn_moverobotcnt++;
}

void FrontierDetectorDMS::unreachablefrontierCallback(const geometry_msgs::PoseStamped::ConstPtr& msg )
{
	ROS_INFO("@unreachablefrontierCallback: The robot is at [%s] state\n ",  robot_state[me_robotstate+1] );

	geometry_msgs::PoseStamped unreachablepose = *msg ;

	appendUnreachablePoint( unreachablepose );

//	for (const auto & di : m_unreachable_frontier_set)
//	    ROS_WARN("unreachable pts: %f %f\n", di.d[0], di.d[1]);

	// stop the robot and restart frontier detection procedure

	ROS_WARN("+++++++++++Canceling the unreachable goal +++++++++++++\n");

	{
		const std::unique_lock<mutex> lock(mutex_robot_state) ;
		me_robotstate = ROBOT_STATE::FORCE_TO_STOP ;
	}

	m_move_client.stopTrackingGoal();
	m_move_client.waitForResult();
	m_move_client.cancelGoal();
	m_move_client.waitForResult();

	ROS_INFO("+++++++++++ Robot is ready for motion +++++++++++++\n");
	{
		const std::unique_lock<mutex> lock(mutex_robot_state) ;
		me_robotstate = ROBOT_STATE::ROBOT_IS_READY_TO_MOVE ;
	}

}


}

