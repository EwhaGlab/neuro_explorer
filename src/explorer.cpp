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

#include "explorer.hpp"

namespace neuroexplorer
{

Explorer::Explorer():
m_worldFrameId("map"), m_baseFrameId("base_footprint"),
mu_cmheight(0), mu_cmwidth(0), me_robotstate(ROBOT_STATE::ROBOT_IS_NOT_MOVING),
m_move_client("move_base", true),
mf_robot_radius(0.3), mb_explorationisdone(false), mn_globalmap_xc_(0), mn_globalmap_yc_(0), mn_rows(0), mn_cols(0)
{

}

Explorer::~Explorer(){}

cv::Point2f Explorer::img2gridmap( cv::Point img_pt ){};
cv::Point Explorer::gridmap2img( cv::Point2f grid_pt ){};

void Explorer::mapdataCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
	ROS_ERROR("this shouldn't be called \n");
}
vector<cv::Point> Explorer::eliminateSupriousFrontiers( nav_msgs::OccupancyGrid &costmapData, vector<cv::Point> frontierCandidates, int winsize = 25)
{}


bool Explorer::correctFrontierPosition( const nav_msgs::OccupancyGrid &gridmap, const cv::Point& frontierCandidate, const int& winsize, cv::Point& correctedPoint  )
{
// pts found in pyrdown sampled map might be slightly off owing to the coarse level analysis
// we need to correct this point. Bring this point to its nearby border line

	correctedPoint = frontierCandidate;

	CV_Assert( winsize % 2 > 0 ); // must be an odd number

	int height = gridmap.info.height ;
	int width  = gridmap.info.width ;
	std::vector<signed char> Data=gridmap.data;

	int w = winsize ;
	int h = winsize ;

	int yc = winsize - (winsize-1)/2 ;
	int xc = winsize - (winsize-1)/2 ;
	int y = yc - 1;
	int x = xc - 1;

	int gy_ = frontierCandidate.y;
	int gx_ = frontierCandidate.x;
	int gx = gx_;
	int gy = gy_;

	//vector<vector<int>> dirs = { {0, -1}, {-1, 0}, {0, 1}, {1, 0} } ;

	int numcells = w * h ;

	int i = 0;
	int curridx = x + y * width ;
	int cnt = 0;

	int idx = gx_ + (gy_ ) * width ;

	int8_t fpt_hat_occupancy = Data[idx] ;

	//ROS_INFO("orig idx: %d (%d,%d) (%d,%d)", idx, (gx_), (gy_), x, y );

	if( fpt_hat_occupancy == 0 ) // is at the free region. thus, the closest unknown cell is the corrected fpt.
	{
		//ROS_INFO("cent occupancy is 0\n");
		while ( cnt < numcells )
		{
			for( int j = (i%2)*2; j < (i%2)*2+2; j++ )
			{
				int dx = nccxidx[j];
				int dy = nccyidx[j];

				for( int k=0; k < i+1; k++ )
				{
					x = x + dx ;
					y = y + dy ;
					if( (0 <= x && x < w ) && (0 <= y && y < h ) )
					{
						gx = gx + dx;
						gy = gy + dy;
						idx = gx + gy * width ;
						int8_t out = Data[idx] ;
						//ROS_INFO(" %d (%d,%d) (%d,%d)", idx, gx, gy, dx, dy );
						if( out == -1 ) // fpt_hat is a free cell. Thus, this pt is the corrected fpt.
						{
							//ROS_INFO(" corrected pixel is %d %d \n", gx, gy );
							correctedPoint.x = gx ;
							correctedPoint.y = gy ;
							return true;
						}
					}
					cnt++ ;
				}
			}
			i++ ;
		}
	}
	// if fpt_hat is already at the unknown region, we might need to shift this position to a boundary cell position
	else if( fpt_hat_occupancy < 0 )
	{
		//ROS_INFO("cent occupancy is -1\n");

		// see if there is a neighboring free cell
		for(int ii=-1; ii <2; ii++ )
		{
			for(int jj=-1; jj <2; jj++ )
			{
				if(ii == 0 && jj == 0)
					continue;

				int8_t nn = Data[ gx + jj + (gy + ii)*width];
				if(nn == 0)
				{
					//ROS_INFO("nn pix %d %d is free, thus no need to do any correction \n", gx+jj, gy+ii);
					return true;
				}
			}
		}

		while ( cnt < numcells )
		{
			for( int j = (i%2)*2; j < (i%2)*2+2; j++ )
			{
				int dx = nccxidx[j];
				int dy = nccyidx[j];

				for( int k=0; k < i+1; k++ )
				{
					x = x + dx ;
					y = y + dy ;
					//ROS_INFO("x y h w i cnt (%d %d) (%d %d) %d %d %d | ", x, y, h, w, j, i, cnt);
					if( (0 <= x && x < w ) && (0 <= y && y < h ) )
					{
						gx = gx + dx;
						gy = gy + dy;
						idx = gx + gy * width ;
						int8_t out = Data[idx] ;
					//	ROS_INFO(" %d (%d,%d) (%d,%d)", idx, gx, gy, dx, dy );

						// ------------ //
						// oooooooooooo //
						// oooo x ooooo //

						if( out == 0 ) // We found the nn (free) border pixel. go ahead check its 7 neighbors
						{
							//ROS_INFO(" found a free pixel at %d %d \n", gx, gy );
							for(int ii=-1; ii <2; ii++ )
							{
								for(int jj=-1; jj <2; jj++ )
								{
									if(ii == 0 && jj == 0)
										continue;

									int8_t nn = Data[ gx + jj + (gy + ii)*width];
									if(nn < 0)
									{
										gx = gx + jj ;
										gy = gy + ii ;
										//ROS_INFO(" corrected pixel is %d %d \n", gx, gy );
										correctedPoint.x = gx ;
										correctedPoint.y = gy ;
										return true;
									}
								}
							}
						}
					}
					cnt++ ;
				}
			}
			i++ ;
		}
	}

	else
	{
		return false ;
	}

	return true ;
}


//void Explorer::SetVizMarker( const string& frame_id,
//					const float& fR, const float& fG, const float& fB, const float& fscale, visualization_msgs::Marker&  viz_marker)
//{
//	viz_marker.header.frame_id= frame_id;
//	viz_marker.header.stamp=ros::Time(0);
//	viz_marker.ns= "markers";
//	viz_marker.id = 0;
//	viz_marker.type = viz_marker.POINTS;
//
//	viz_marker.action = viz_marker.ADD;
//	viz_marker.pose.orientation.w =1.0;
//	viz_marker.scale.x= fscale;
//	viz_marker.scale.y= fscale;
//
//	viz_marker.color.r = fR;
//	viz_marker.color.g = fG;
//	viz_marker.color.b = fB;
//	viz_marker.color.a=1.0;
//	viz_marker.lifetime = ros::Duration();
//}
//
//void Explorer::SetVizMarkersArray( const string& frame_id,
//					const float& fR, const float& fG, const float& fB, const float& fscale, visualization_msgs::Marker&  viz_marker)
//{
//	viz_marker.header.frame_id= frame_id;
//	viz_marker.header.stamp=ros::Time(0);
//	viz_marker.ns= "markers";
//	viz_marker.id = 0;
//	viz_marker.type = viz_marker.POINTS;
//
//	viz_marker.action = viz_marker.ADD;
//	viz_marker.pose.orientation.w =1.0;
//	viz_marker.scale.x= fscale;
//	viz_marker.scale.y= fscale;
//
//	viz_marker.color.r = fR;
//	viz_marker.color.g = fG;
//	viz_marker.color.b = fB;
//	viz_marker.color.a=1.0;
//	viz_marker.lifetime = ros::Duration();
//}

//void accessFrontierPoint( ){}

void Explorer::saveMetaData(const string& metadatafilename, const nav_msgs::MapMetaData& mapInfo, const geometry_msgs::PoseWithCovarianceStamped& rpos_w )
{
	ofstream ofs_metadata(metadatafilename) ;
	ofs_metadata << rpos_w.pose.pose.position.x << " " << rpos_w.pose.pose.position.y << " "
				 << mapInfo.height << " " << mapInfo.width << " " << mapInfo.origin.position.x << " " << mapInfo.origin.position.y << mapInfo.resolution << endl;
	ofs_metadata.close();
}

void Explorer::saveRobotPose(const string& rposefilename, const geometry_msgs::PoseWithCovarianceStamped& rpos_w )
{
	ofstream ofs_rpose(rposefilename) ;
	ofs_rpose << rpos_w.pose.pose.position.x << " " << rpos_w.pose.pose.position.y << endl;
	ofs_rpose.close();
}

void Explorer::saveGridmap( const string& mapfilename, const string& mapinfofilename, const nav_msgs::OccupancyGrid& mapData )
{
	ofstream ofs_map(mapfilename) ;
	ofstream ofs_info(mapinfofilename);

	int height = mapData.info.height ;
	int width  = mapData.info.width ;
	float origx = mapData.info.origin.position.x ;
	float origy = mapData.info.origin.position.y ;
	float resolution = mapData.info.resolution ;

	std::vector<signed char> Data=mapData.data;
	ofs_info << origx << " " << origy << " " << width << " " << height << " " << resolution << endl;;
	ofs_info.close();

	for(int ridx = 0; ridx < height; ridx++)
	{
		for(int cidx = 0; cidx < width; cidx++)
		{
			int value = static_cast<int>( Data[ridx * width + cidx] ) ;
			ofs_map << value << " ";
		}
		ofs_map << "\n";
	}
	ofs_map.close();
}

void Explorer::saveGridmap( const string& mapfilename, const nav_msgs::OccupancyGrid& mapData )
{
	ofstream ofs_map(mapfilename) ;
	int height = mapData.info.height ;
	int width  = mapData.info.width ;
	float origx = mapData.info.origin.position.x ;
	float origy = mapData.info.origin.position.y ;
	float resolution = mapData.info.resolution ;

	std::vector<signed char> Data=mapData.data;
	//ofs_map << width << " " << height << " " << origx << " " << origy << " " << resolution;
	for(int ridx = 0; ridx < height; ridx++)
	{
		for(int cidx = 0; cidx < width; cidx++)
		{
			int value = static_cast<int>( Data[ridx * width + cidx] ) ;
			ofs_map << value << " ";
		}
		ofs_map << "\n";
	}
	ofs_map.close();
}

void Explorer::writeGridmapToPNG( const string& filename, const nav_msgs::OccupancyGrid &mapData )
{
//	ofstream ofs_map(filename) ;
//	int height = mapData.info.height ;
//	int width  = mapData.info.width ;
//	float origx = mapData.info.origin.position.x ;
//	float origy = mapData.info.origin.position.y ;
//	float resolution = mapData.info.resolution ;
//
//	std::vector<signed char> Data=mapData.data;
//	//ofs_map << width << " " << height << " " << origx << " " << origy << " " << resolution;
//	for(int ridx = 0; ridx < height; ridx++)
//	{
//		for(int cidx = 0; cidx < width; cidx++)
//		{
//			int value = static_cast<int>( Data[ridx * width + cidx] ) ;
//			ofs_map << value << " ";
//		}
//		ofs_map << "\n";
//	}
//	ofs_map.close();
}

void Explorer::saveFrontierCandidates( const string& filename, const vector<FrontierPoint>& voFrontierCandidates )
{
	ofstream ofs_fpts(filename) ;
	for(size_t idx=0; idx < voFrontierCandidates.size(); idx++)
	{
		FrontierPoint oFP = voFrontierCandidates[idx];
		cv::Point initposition = oFP.GetInitGridmapPosition() ;
		cv::Point correctedposition = oFP.GetCorrectedGridmapPosition() ;
		float fcmconf = oFP.GetCMConfidence() ;
		float fgmconf = oFP.GetGMConfidence() ;
		ofs_fpts << fcmconf << " " << fgmconf << " " << oFP.isConfidentFrontierPoint() << " " <<
		initposition.x << " " << initposition.y << " " << correctedposition.x << " " << correctedposition.y << endl;
	}
	ofs_fpts.close();
}


}

