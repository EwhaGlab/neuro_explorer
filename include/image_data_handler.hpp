/*
 * image_data_handler.hpp
 *
 *  Created on: Oct 19, 2023
 *      Author: hankm
 */

#ifndef INCLUDE_IMAGE_DATA_HANDLER_HPP_
#define INCLUDE_IMAGE_DATA_HANDLER_HPP_

#include <fstream>
#include <ros/console.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "octomap_server/mapframedata.h"

#include <ros/ros.h>
#include <ros/console.h>
#include "nav_msgs/OccupancyGrid.h"
#include "frontier_point.hpp"

namespace neuroexplorer
{

//typedef struct pointclass
//{
//public:
//	pointclass( int nx, int ny, int nlabel ):
//		x(nx), y(ny), label(nlabel)
//	{};
//	int x, y, label;
//}PointClass;
//
//typedef struct rgb
//{
//public:
//	rgb( float fr, float fg, float fb ):
//		r(fr), g(fg), b(fb)
//	{};
//	float r, g, b;
//}rgb;
//
//typedef struct pointclassset
//{
//
//public:
//
//	pointclassset( rgb cvminColor, rgb cvmaxcolor, int nmaxlabel ):
//		minColor(cvminColor), maxColor(cvmaxcolor), maxlabel(nmaxlabel)
//	{};
//	inline rgb get_color(double alpha, const rgb& c0, const rgb& c1)
//	{
//		float r = (1-alpha) * c0.r  +  alpha * c1.r ;
//		float g = (1-alpha) * c0.g  +  alpha * c1.g ;
//		float b = (1-alpha) * c0.b  +  alpha * c1.b ;
//	    return rgb(r,g,b);
//	}
//	vector<PointClass> point_classes ;
//	vector<rgb> point_colors ;
//
//	void push_point( PointClass pc)
//	{
//		point_classes.push_back(pc);
//		float alpha = (float)pc.label / (float)maxlabel ;
//		rgb color = get_color( alpha, minColor, maxColor );
//		point_colors.push_back(color);
//	}
//
//	vector<PointClass> GetPointClass() const {return point_classes; }
//	vector<rgb> GetPointColor() const {return point_colors; }
//
//	int maxlabel ;
//	rgb minColor, maxColor ;
//
//}PointClassSet;

class ImageDataHandler
{
public:

	ImageDataHandler(){};
	ImageDataHandler( int nheight, int nwidth, float fsigma);
	virtual ~ImageDataHandler();

	void transform_map_to_robotposition( const cv::Mat& input_map, const int& rx_gm, const int& ry_gm, const uint8_t& base_val, cv::Mat& out_map );
	void inv_transform_map_to_robotposition( const cv::Mat& input_map, const int& rx_gm, const int& ry_gm, const uint8_t& base_val, cv::Mat& out_map ) ;
	void inv_transform_point_to_robotposition( const vector<PointClass>& pts, const int& nheight, const int& nwidth, const int& rx_gm, const int& ry_gm, vector<PointClass>& pts_tformed ) ;

	void generate_gaussian_image( const int& nheight, const int& nwidth, const int& kernel_size, const float& sigma  );
	void obsmap_from_gmap( const cv::Mat& gmap, cv::Mat& obs_map) ;
	void generate_astar_net_input( const cv::Mat& fr_img_8u, const cv::Mat& gmap_8u, const cv::Mat& gauss_img_32f, cv::Mat& out_map_32f  );
	void generate_viz_net_input( const cv::Mat& fr_img_8u, const cv::Mat& gmap_8u, cv::Mat& out_map_32f  );

	cv::Mat GetGaussianImg() const {return mcv_gaussimg_32f; };

	inline cv::Scalar get_colour(double alpha, const cv::Scalar& c0, const cv::Scalar& c1)
	{
	    return (1-alpha) * c0  +  alpha * c1;
	}

	inline cv::Scalar get_colour(double v, double vmin, double vmax)
	{
	    // clamp value within range
	    v = v <= vmin ? vmin
	        : v >= vmax ? vmax
	        : v;

	    const double alpha = (vmax <= vmin)
	        ? 0.5                   // avoid divide by zero
	        : (v - vmin) / (vmax - vmin);

	    static const cv::Scalar blue{ 255, 0, 0 };
	    static const cv::Scalar white{ 255, 255, 255 };
	    static const cv::Scalar red{ 0, 0, 255 };

	    if (alpha < 0.5) {
	        return get_colour(alpha * 2, blue, white);
	    } else {
	        return get_colour(alpha * 2 - 1, white, red);
	    }
	}

private:

	cv::Mat mcv_gaussimg_32f ;
	int mn_height, mn_width, mn_kernel_size ;
	float mf_sigma;
};

}

#endif /* INCLUDE_IMAGE_DATA_HANDLER_HPP_ */
