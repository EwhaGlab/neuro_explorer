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


#ifndef INCLUDE_FRONTIER_POINT_HPP_
#define INCLUDE_FRONTIER_POINT_HPP_

#include <ros/console.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "octomap_server/mapframedata.h"
#include <algorithm>
#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include "nav_msgs/OccupancyGrid.h"
#include "dffp.hpp"

namespace neuroexplorer
{

struct pointset
{
	std::pair<float, float> xy;
	pointset( float x_ =0.f, float y_ = 0.f) : xy(std::make_pair(x_, y_)){ p[0] = xy.first; p[1] = xy.second; }
	bool operator<(const pointset& rhs) const
	{return xy < rhs.xy;}

	float p[2]; // point in world coordinate frame
//	int label;
};

typedef struct pointclass
{
public:
	pointclass( int nx, int ny, int nlabel ):
		x(nx), y(ny), label(nlabel)
	{};
	int x, y, label;
}PointClass;

typedef struct rgb
{
public:
	rgb( float fr, float fg, float fb ):
		r(fr), g(fg), b(fb)
	{};
	float r, g, b;
}rgb;

typedef struct pointclassset
{

public:

	pointclassset( rgb cvminColor, rgb cvmaxcolor, int nmaxlabel ):
		minColor(cvminColor), maxColor(cvmaxcolor), maxlabel(nmaxlabel)
	{};
	inline rgb get_color(double alpha, const rgb& c0, const rgb& c1)
	{
		float r = (1-alpha) * c0.r  +  alpha * c1.r ;
		float g = (1-alpha) * c0.g  +  alpha * c1.g ;
		float b = (1-alpha) * c0.b  +  alpha * c1.b ;
	    return rgb(r,g,b);
	}

	void push_point( PointClass pc)
	{
		point_classes.push_back(pc);
		float alpha = (float)pc.label / (float)maxlabel ;
		rgb color = get_color( alpha, minColor, maxColor );
		point_colors.push_back(color);
	}

	std::vector<PointClass> GetPointClass() const {return point_classes; }
	std::vector<rgb> GetPointColor() const {return point_colors; }
	int size() const {return point_colors.size(); }

private:
	std::vector<PointClass> point_classes ;
	std::vector<rgb> point_colors ;
	int maxlabel ;
	rgb minColor, maxColor ;

}PointClassSet;


enum PointState{ NOT_TESTED = 0, GM_TESTED, CM_FILTERED, FULL_TESTED };


class FrontierPoint
{
public:
	FrontierPoint( cv::Point FrontierCandidate_dsn, int mapheight, int mapwidth,
			float fmaporigy, float fmaporigx, float fmapres, int num_pyrdown): //, float fgridmap_conf_thr, float fcostmap_conf_thr ):
	mf_gridmap_confidence(1.f), mf_costmap_confidence(1.f),
	//m_bcostmap_consent(true), m_bgridmap_consent(true),
	mn_initposition_dsgm( FrontierCandidate_dsn ), // dsgm refers to n down-sampled gridmap image
	mb_isfrontierpoint(true), mb_isreachable(true),
	me_pointstate( PointState::NOT_TESTED ),
	//m_fgridmap_conf_thr(fgridmap_conf_thr), m_fcostmap_conf_thr(fcostmap_conf_thr),
	mn_height(mapheight), mn_width(mapwidth),
	mf_origx(fmaporigx), mf_origy(fmaporigy),
	mf_res(fmapres), m_nNumPyrDownSample(num_pyrdown)
	{
		//mn_globalcentx = mn_width / 2 ;
		//mn_globalcenty = mn_height/ 2 ;
		//SetWorldCoordiate();
		m_nScale = pow(2, m_nNumPyrDownSample) ;

		mn_initposition_gm 	   =	mn_initposition_dsgm * m_nScale ;
		mn_correctedposition_gm = 	mn_initposition_gm ;

		SetInitWorldCoordiate( );
		SetCorrectedCoordinate( mn_initposition_gm );
	}

	~FrontierPoint(){} ;

	void SetFrontierRegion( std::vector<cv::Point> inpoints ){ mn_frontier_region = inpoints; }
	void SetCostmapConfidence( float fconfidence){ mf_costmap_confidence = fconfidence ; }
	void SetGridmapConfidence( float fconfidence){ mf_gridmap_confidence = fconfidence ; }
	float GetCMConfidence() const { return mf_costmap_confidence; }
	float GetGMConfidence() const { return mf_gridmap_confidence; }

	cv::Point world2gridmap( cv::Point2f point_w)
	{
		int ngmx = static_cast<int>( (point_w.x - mf_origx) / mf_res ) ;
		int ngmy = static_cast<int>( (point_w.y - mf_origy) / mf_res ) ;
		return cv::Point( ngmx, ngmy );
	}

	cv::Point2f gridmap2world( cv::Point img_pt_ds0 )
	{
		float fgmx = static_cast<float>(img_pt_ds0.x) ;
		float fgmy = static_cast<float>(img_pt_ds0.y) ;

		float fgx = ( fgmx * mf_res ) + mf_origx ; // + fXstart ) ;
		float fgy = ( fgmy * mf_res ) + mf_origy ; // - fYstart ) ;
		return cv::Point2f( fgx, fgy );
	}

	// given frontier input convert it to world coord
	void SetInitWorldCoordiate( )
	{
		mf_initposition_w = gridmap2world( mn_initposition_gm  );
	}

	void SetCorrectedCoordinate( const cv::Point& pt )
	{
		mn_correctedposition_gm = pt;
		mf_correctedposition_w = gridmap2world( pt );
	}

	void SetFrontierFlag( const float& fcm_conf, const float& fgm_conf, const bool& bisexplored, const bool& bisfrontier )
	{
		if( mf_gridmap_confidence < fgm_conf || mf_costmap_confidence < fcm_conf || bisexplored || !bisfrontier)
			mb_isfrontierpoint = false;
		else
			mb_isfrontierpoint = true;
	}

	void SetFrontierFlag( const float& fcm_conf, const float& fgm_conf)
	{
		if( mf_gridmap_confidence < fgm_conf || mf_costmap_confidence < fcm_conf)
			mb_isfrontierpoint = false;
		else
			mb_isfrontierpoint = true;
	}

	void SetFrontierFlag( const bool& bflag )
	{
		mb_isfrontierpoint = bflag;
	}

	void SetReachability( const bool& bisreachable )
	{
		mb_isreachable = bisreachable;
	}

	bool isConfidentFrontierPoint() const { return mb_isfrontierpoint; }
	bool isReachable() const { return mb_isreachable; }

	cv::Point GetDSGridmapPosition()		const {return mn_initposition_dsgm; } // returns input dsgm position
	cv::Point GetInitGridmapPosition()		const {return mn_initposition_gm; };
	cv::Point2f GetInitWorldPosition()		const {return mf_initposition_w; };
	cv::Point GetCorrectedGridmapPosition() const {return mn_correctedposition_gm; };
	cv::Point2f GetCorrectedWorldPosition() const {return mf_correctedposition_w; };
	int GetNumDS()							const {return m_nNumPyrDownSample; };

	void saveFrontierInfo( std::string strfilename )
	{
		std::ofstream ofs( strfilename ) ;

		ofs << mn_initposition_gm.x << " " << mn_initposition_gm.y << std::endl;
		ofs << mf_initposition_w.x  << " " << mf_initposition_w.y  << std::endl;
		ofs << "\n" ;

		for( int idx=0; idx < mn_frontier_region.size(); idx++ )
		{
			ofs << mn_frontier_region[idx].x << " " << mn_frontier_region[idx].y << std::endl;
		}
		ofs.close();
	}

//	bool operator< (const FrontierPoint& other) const
//	{
//		return mf_pot_value > other.mf_pot_value ;
//	}

	float mf_vizvalue ;
	float mf_potvalue ; // value of the point in terms of (Astar Pot or Ensembled score )
	float mf_ensembled_value ;

private:

	//bool m_bcostmap_consent, m_bgridmap_consent;
	float mf_gridmap_confidence ;
	float mf_costmap_confidence ;
	float mf_gridmap_conf_thr ;
	float mf_costmap_conf_thr ;

	cv::Point2f mf_initposition_w ;			// on world (meter) coord
	cv::Point2f mf_correctedposition_w ;		// on world (meter) coord

	cv::Point mn_initposition_dsgm ;			// on downsampled image (gm/cm) coord. --i.e) smaller than the orig size
	cv::Point mn_initposition_gm ; 			// on image (gm/cm) coord ( ex. 4K x 4K original size )
	cv::Point mn_correctedposition_gm ;		// on image (gm/cm) coord
	int8_t me_pointstate;

	std::vector<cv::Point> mn_frontier_region ;

	bool mb_isfrontierpoint ;
	bool mb_isreachable ;
	bool mb_isptcovered ;
	int mn_height;
	int mn_width;
	int mn_globalcentx ;
	int mn_globalcenty ;
	float mf_origx ;
	float mf_origy ;
	float mf_res ;
	int m_nNumPyrDownSample ;
	int m_nScale ;
};

struct sort_by_potvalue
{
	inline bool operator() ( const FrontierPoint& fpt1, const FrontierPoint& fpt2 )
	{
		return ( fpt1.mf_potvalue > fpt2.mf_potvalue ) ;
	}
};

struct sort_by_vizvalue
{
	inline bool operator() ( const FrontierPoint& fpt1, const FrontierPoint& fpt2 )
	{
		return ( fpt1.mf_vizvalue > fpt2.mf_vizvalue ) ;
	}
};

struct sort_by_ensembled_value
{
	inline bool operator() ( const FrontierPoint& fpt1, const FrontierPoint& fpt2 )
	{
		return ( fpt1.mf_ensembled_value > fpt2.mf_ensembled_value ) ;
	}
};

}

#endif /* INCLUDE_FRONTIER_POINT_HPP_ */
