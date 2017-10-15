///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#include <Face_utils.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
namespace FaceAnalysis
{

	// Pick only the more stable/rigid points under changes of expression
	void extract_rigid_points(cv::Mat_<double>& source_points, cv::Mat_<double>& destination_points)
	{
		if(source_points.rows == 68)
		{
			cv::Mat_<double> tmp_source = source_points.clone();
			source_points = cv::Mat_<double>();

			// Push back the rigid points (some face outline, eyes, and nose)
			source_points.push_back(tmp_source.row(1));
			source_points.push_back(tmp_source.row(2));
			source_points.push_back(tmp_source.row(3));
			source_points.push_back(tmp_source.row(4));
			source_points.push_back(tmp_source.row(12));
			source_points.push_back(tmp_source.row(13));
			source_points.push_back(tmp_source.row(14));
			source_points.push_back(tmp_source.row(15));
			source_points.push_back(tmp_source.row(27));
			source_points.push_back(tmp_source.row(28));
			source_points.push_back(tmp_source.row(29));
			source_points.push_back(tmp_source.row(31));
			source_points.push_back(tmp_source.row(32));
			source_points.push_back(tmp_source.row(33));
			source_points.push_back(tmp_source.row(34));
			source_points.push_back(tmp_source.row(35));
			source_points.push_back(tmp_source.row(36));
			source_points.push_back(tmp_source.row(39));
			source_points.push_back(tmp_source.row(40));
			source_points.push_back(tmp_source.row(41));
			source_points.push_back(tmp_source.row(42));
			source_points.push_back(tmp_source.row(45));
			source_points.push_back(tmp_source.row(46));
			source_points.push_back(tmp_source.row(47));

			cv::Mat_<double> tmp_dest = destination_points.clone();
			destination_points = cv::Mat_<double>();

			// Push back the rigid points
			destination_points.push_back(tmp_dest.row(1));
			destination_points.push_back(tmp_dest.row(2));
			destination_points.push_back(tmp_dest.row(3));
			destination_points.push_back(tmp_dest.row(4));
			destination_points.push_back(tmp_dest.row(12));
			destination_points.push_back(tmp_dest.row(13));
			destination_points.push_back(tmp_dest.row(14));
			destination_points.push_back(tmp_dest.row(15));
			destination_points.push_back(tmp_dest.row(27));
			destination_points.push_back(tmp_dest.row(28));
			destination_points.push_back(tmp_dest.row(29));
			destination_points.push_back(tmp_dest.row(31));
			destination_points.push_back(tmp_dest.row(32));
			destination_points.push_back(tmp_dest.row(33));
			destination_points.push_back(tmp_dest.row(34));
			destination_points.push_back(tmp_dest.row(35));
			destination_points.push_back(tmp_dest.row(36));
			destination_points.push_back(tmp_dest.row(39));
			destination_points.push_back(tmp_dest.row(40));
			destination_points.push_back(tmp_dest.row(41));
			destination_points.push_back(tmp_dest.row(42));
			destination_points.push_back(tmp_dest.row(45));
			destination_points.push_back(tmp_dest.row(46));
			destination_points.push_back(tmp_dest.row(47));
		}
	}

	// Aligning a face to a common reference frame
	void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame, const LandmarkDetector::CLNF& clnf_model, bool rigid, double sim_scale, int out_width, int out_height)
	{
		// Will warp to scaled mean shape
		cv::Mat_<double> similarity_normalised_shape = clnf_model.pdm.mean_shape * sim_scale;
	
		// Discard the z component
		similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

		cv::Mat_<double> source_landmarks = clnf_model.detected_landmarks.reshape(1, 2).t();
		cv::Mat_<double> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

		// Aligning only the more rigid points
		if(rigid)
		{
			extract_rigid_points(source_landmarks, destination_landmarks);
		}

		cv::Matx22d scale_rot_matrix = LandmarkDetector::AlignShapesWithScale(source_landmarks, destination_landmarks);
		cv::Matx23d warp_matrix;

		warp_matrix(0,0) = scale_rot_matrix(0,0);
		warp_matrix(0,1) = scale_rot_matrix(0,1);
		warp_matrix(1,0) = scale_rot_matrix(1,0);
		warp_matrix(1,1) = scale_rot_matrix(1,1);

		double tx = clnf_model.params_global[4];
		double ty = clnf_model.params_global[5];

		cv::Vec2d T(tx, ty);
		T = scale_rot_matrix * T;

		// Make sure centering is correct
		warp_matrix(0,2) = -T(0) + out_width/2;
		warp_matrix(1,2) = -T(1) + out_height/2;

		cv::warpAffine(frame, aligned_face, warp_matrix, cv::Size(out_width, out_height), cv::INTER_LINEAR);
	}


}