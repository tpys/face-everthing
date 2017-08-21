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
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

#include "GazeEstimation.h"

using namespace std;

using namespace FaceAnalysis;

// For subpixel accuracy drawing
const int gaze_draw_shiftbits = 4;
const int gaze_draw_multiplier = 1 << 4;

cv::Point3f RaySphereIntersect(cv::Point3f rayOrigin, cv::Point3f rayDir, cv::Point3f sphereOrigin, float sphereRadius){

	float dx = rayDir.x;
	float dy = rayDir.y;
	float dz = rayDir.z;
	float x0 = rayOrigin.x;
	float y0 = rayOrigin.y;
	float z0 = rayOrigin.z;
	float cx = sphereOrigin.x;
	float cy = sphereOrigin.y;
	float cz = sphereOrigin.z;
	float r = sphereRadius;

	float a = dx*dx + dy*dy + dz*dz;
	float b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz);
	float c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - r*r;

	float disc = b*b - 4*a*c;

	float t = (-b - sqrt(b*b - 4*a*c))/2*a;

	// This implies that the lines did not intersect, point straight ahead
	if (b*b - 4 * a*c < 0)
		return cv::Point3f(0, 0, -1);

	return rayOrigin + rayDir * t;
}

cv::Point3f GetPupilPosition(cv::Mat_<double> eyeLdmks3d){
	
	eyeLdmks3d = eyeLdmks3d.t();

	cv::Mat_<double> irisLdmks3d = eyeLdmks3d.rowRange(0,8);

	cv::Point3f p (mean(irisLdmks3d.col(0))[0], mean(irisLdmks3d.col(1))[0], mean(irisLdmks3d.col(2))[0]);
	return p;
}

void FaceAnalysis::EstimateGaze(const LandmarkDetector::CLNF& clnf_model, cv::Point3f& gaze_absolute, float fx, float fy, float cx, float cy, bool left_eye)
{
	cv::Vec6d headPose = LandmarkDetector::GetPoseCamera(clnf_model, fx, fy, cx, cy);
	cv::Vec3d eulerAngles(headPose(3), headPose(4), headPose(5));
	cv::Matx33d rotMat = LandmarkDetector::Euler2RotationMatrix(eulerAngles);

	int part = -1;
	for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
	{
		if (left_eye && clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0)
		{
			part = i;
		}
		if (!left_eye && clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
		{
			part = i;
		}
	}

	if (part == -1)
	{
		std::cout << "Couldn't find the eye model, something wrong" << std::endl;
	}

	cv::Mat eyeLdmks3d = clnf_model.hierarchical_models[part].GetShape(fx, fy, cx, cy);

	cv::Point3f pupil = GetPupilPosition(eyeLdmks3d);
	cv::Point3f rayDir = pupil / norm(pupil);

	cv::Mat faceLdmks3d = clnf_model.GetShape(fx, fy, cx, cy);
	faceLdmks3d = faceLdmks3d.t();
	cv::Mat offset = (cv::Mat_<double>(3, 1) << 0, -3.50, 0);
	int eyeIdx = 1;
	if (left_eye)
	{
		eyeIdx = 0;
	}

	cv::Mat eyeballCentreMat = (faceLdmks3d.row(36+eyeIdx*6) + faceLdmks3d.row(39+eyeIdx*6))/2.0f + (cv::Mat(rotMat)*offset).t();

	cv::Point3f eyeballCentre = cv::Point3f(eyeballCentreMat);

	cv::Point3f gazeVecAxis = RaySphereIntersect(cv::Point3f(0,0,0), rayDir, eyeballCentre, 12) - eyeballCentre;
	
	gaze_absolute = gazeVecAxis / norm(gazeVecAxis);
}


void FaceAnalysis::DrawGaze(cv::Mat img, const LandmarkDetector::CLNF& clnf_model, cv::Point3f gazeVecAxisLeft, cv::Point3f gazeVecAxisRight, float fx, float fy, float cx, float cy)
{

	cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

	int part_left = -1;
	int part_right = -1;
	for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
	{
		if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0)
		{
			part_left = i;
		}
		if (clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
		{
			part_right = i;
		}
	}

	cv::Mat eyeLdmks3d_left = clnf_model.hierarchical_models[part_left].GetShape(fx, fy, cx, cy);
	cv::Point3f pupil_left = GetPupilPosition(eyeLdmks3d_left);

	cv::Mat eyeLdmks3d_right = clnf_model.hierarchical_models[part_right].GetShape(fx, fy, cx, cy);
	cv::Point3f pupil_right = GetPupilPosition(eyeLdmks3d_right);

	vector<cv::Point3d> points_left;
	points_left.push_back(cv::Point3d(pupil_left));
	points_left.push_back(cv::Point3d(pupil_left + gazeVecAxisLeft*50.0));

	vector<cv::Point3d> points_right;
	points_right.push_back(cv::Point3d(pupil_right));
	points_right.push_back(cv::Point3d(pupil_right + gazeVecAxisRight*50.0));

	cv::Mat_<double> proj_points;
	cv::Mat_<double> mesh_0 = (cv::Mat_<double>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
	LandmarkDetector::Project(proj_points, mesh_0, fx, fy, cx, cy);
	cv::line(img, cv::Point(cvRound(proj_points.at<double>(0,0) * (double)gaze_draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)gaze_draw_multiplier)),
		cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)gaze_draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)gaze_draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, gaze_draw_shiftbits);

	cv::Mat_<double> mesh_1 = (cv::Mat_<double>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
	LandmarkDetector::Project(proj_points, mesh_1, fx, fy, cx, cy);
	cv::line(img, cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)gaze_draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)gaze_draw_multiplier)),
		cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)gaze_draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)gaze_draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, gaze_draw_shiftbits);
}