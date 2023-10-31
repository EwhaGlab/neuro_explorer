/*
 * image_data_handler.cpp
 *
 *  Created on: Oct 19, 2023
 *      Author: hankm
 */



#include "image_data_handler.hpp"

namespace autoexplorer
{

ImageDataHandler::ImageDataHandler( int nheight, int nwidth, float fsigma = 0.5f):
mn_height(nheight), mn_width(nwidth),
mf_sigma(fsigma)
{
	mn_kernel_size = ( mn_width ) / 4 - 1; // 127 if 512 size image
	mcv_gaussimg_32f = cv::Mat::zeros(mn_height, mn_width, CV_32FC1);
	generate_gaussian_image( mn_height, mn_width, mn_kernel_size, mf_sigma );
};

ImageDataHandler::~ImageDataHandler()
{

}

void ImageDataHandler::transform_map_to_robotposition( const cv::Mat& input_map, const int& rx_gm, const int& ry_gm, const uint8_t& base_val, cv::Mat& out_map )
{
	int nheight = input_map.rows ;
	int nwidth  = input_map.cols ;
	int nchannels	= input_map.channels() ;
	CV_Assert(nchannels <= 3); // I can handle only single or 3 ch images

	int nhalfsize = nheight / 2 ;
	cv::Mat map_expanded = cv::Mat( nheight * 3, nwidth * 3, CV_8UC(nchannels), cv::Scalar::all(base_val) );
	cv::Rect roi(nwidth, nheight, nwidth, nheight );

	cv::Mat map_expanded_roi = map_expanded(roi) ;
	input_map.copyTo(map_expanded_roi) ;

	int ys = int( nheight + ry_gm - nhalfsize ) ;
	int xs = int( nwidth + rx_gm - nhalfsize ) ;
	cv::Rect crop_roi(xs, ys, nwidth, nheight );
	out_map = map_expanded(crop_roi).clone() ;
}

void ImageDataHandler::inv_transform_map_to_robotposition( const cv::Mat& input_map, const int& rx_gm, const int& ry_gm, const uint8_t& base_val, cv::Mat& out_map )
{
	int nheight = input_map.rows ;
	int nwidth  = input_map.cols ;
	int nchannels	= input_map.channels() ;
	CV_Assert(nchannels <= 3); // I can handle only single or 3 ch images

	int nhalfsize = nheight / 2 ;
	cv::Mat map_expanded = cv::Mat( nheight * 3, nwidth * 3, CV_8UC(nchannels), cv::Scalar::all(base_val) );
	cv::Rect roi(nwidth, nheight, nwidth, nheight );

	cv::Mat map_expanded_roi = map_expanded(roi) ;
	input_map.copyTo(map_expanded_roi) ;
	int ys = nheight + nhalfsize - ry_gm  ;
	int xs = nwidth  + nhalfsize - rx_gm ;

	cv::Rect crop_roi(xs, ys, nwidth, nheight );
	out_map = map_expanded(crop_roi).clone() ;
}


void ImageDataHandler::inv_transform_point_to_robotposition( const vector<PointClass>& pts, const int& nheight,
		const int& nwidth, const int& rx_gm, const int& ry_gm, vector<PointClass>& pts_tformed )
{
	int centx = nwidth / 2 ;
	int centy = nheight / 2 ;
	int xoffset = rx_gm - centx ;
	int yoffset = ry_gm - centy ;

	for( int idx=0; idx < pts.size(); idx++)
	{
		PointClass pt_t( pts[idx].x + xoffset, pts[idx].y + yoffset, pts[idx].label );
		pts_tformed.push_back(pt_t);
	}
}


void ImageDataHandler::generate_gaussian_image( const int& nheight, const int& nwidth, const int& kernel_size, const float& sigma )
{
	CV_Assert(nheight > kernel_size && nwidth > kernel_size );
	CV_Assert( kernel_size % 2 == 1 );

    cv::Mat GKernel  = cv::Mat::zeros( kernel_size, kernel_size, CV_32FC1 );
    vector<float> ys, xs;
	int N = (kernel_size - 1) / 2 ;
	float step_size = 1.f / float(N) ;
	float fstep = -1.f ;
    for(int idx=0; idx < kernel_size; idx++ )
    {
    	ys.push_back(fstep) ;
    	xs.push_back(fstep) ;
    	fstep += step_size ;
    }

    float r = 0;
    float s = 2.0 * sigma * sigma;

    // generating NxN kernel
    for (int y = 0; y < ys.size(); y++)
    {
        for (int x = 0; x < xs.size(); x++)
        {
            r = sqrt(xs[x] * xs[x] + ys[y] * ys[y]);
            GKernel.at<float>(y, x) = (exp(-(r * r) / s)) / ( 2* M_PI * sigma );
        }
    }
    float maxG = GKernel.at<float>(N, N) ;
    GKernel = GKernel / maxG ;

//cv::normalize( gkernel2d, gkernel2d, 1, 0, cv::NORM_L2, -1);
//ROS_INFO("gkernel: %d %d \n", GKernel.rows, GKernel.cols );
//cv::Mat gkernel2d_tmp = GKernel;
//GKernel.convertTo(gkernel2d_tmp, CV_8UC1, 255.0);
//cv::imwrite("/home/hankm/results/neuro_exploration_res/gkernel2d.png", gkernel2d_tmp );

	int nxoffset = nwidth /2 - (kernel_size - 1) / 2 ;
	int nyoffset = nheight/2 - (kernel_size - 1) / 2 ;
	cv::Rect roi( nxoffset, nyoffset, kernel_size, kernel_size );

	cv::Mat tmp;
	tmp = mcv_gaussimg_32f(roi);
	GKernel.copyTo( tmp );
}

void ImageDataHandler::obsmap_from_gmap( const cv::Mat& gmap, cv::Mat& obs_map)
{
	obs_map = gmap.clone();
	cv::threshold(gmap, obs_map, 200, 1.f, CV_32F);
}

void ImageDataHandler::generate_astar_net_input( const cv::Mat& fr_img_8u, const cv::Mat& gmap_8u, const cv::Mat& gauss_img_32f, cv::Mat& out_map_32f )
{
// fr_img and gmap must be tformed to center !!
// fr_img and gmap must be 32F  (depth() == 5)
// That is, the center of each map must correspond to robot's position
	int nheight = fr_img_8u.rows ;
	int nwidth  = fr_img_8u.cols ;
	cv::Mat obs_map_32f = cv::Mat::zeros(nheight, nwidth, CV_32FC1 );
	cv::Mat fr_map_32f = cv::Mat::zeros(nheight, nwidth, CV_32FC1 );
	cv::Mat tmp_map_32f = cv::Mat::zeros(nheight, nwidth, CV_32FC3 );
	out_map_32f = cv::Mat::zeros(nheight, nwidth, CV_32FC3 );

	fr_img_8u.convertTo(fr_map_32f, CV_32F, 1.f/255.f);
	cv::Mat obs_map_8u ;

	cv::threshold(gmap_8u, obs_map_8u, 254, 255, cv::THRESH_BINARY) ;
	obs_map_8u.convertTo(obs_map_32f, CV_32F, 1.f/255.f);

	vector<cv::Mat> out_map_chs;
	out_map_chs.push_back(fr_map_32f);
	out_map_chs.push_back(obs_map_32f);
	out_map_chs.push_back(gauss_img_32f);

	cv::merge(out_map_chs, out_map_32f);

}



}

