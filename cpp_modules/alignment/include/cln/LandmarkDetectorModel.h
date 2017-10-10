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

#ifndef __LANDMARK_DETECTOR_MODEL_h_
#define __LANDMARK_DETECTOR_MODEL_h_

// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <memory>

#include "PDM.h"
#include "Patch_experts.h"
#include "LandmarkDetectionValidator.h"
#include "LandmarkDetectorParameters.h"
#include "cl_common.h"
#include "cl_detector.h"
#include "cl_alignment.h"

using namespace std;

namespace LandmarkDetector
{

// A main class containing all the modules required for landmark detection
// Face shape model
// Patch experts
// Optimization techniques
class CLNF{

public:
	//===========================================================================
	// Member variables that contain the model description

	// The linear 3D Point Distribution Model
    PDM					pdm;
	// The set of patch experts
	Patch_experts		patch_experts;

	// The local and global parameters describing the current model instance (current landmark detections)

	// Local parameters describing the non-rigid shape
	cv::Mat_<double>    params_local;

	// Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
	cv::Vec6d           params_global;

	// A collection of hierarchical CLNF example that can be used for refinement
	vector<CLNF>					hierarchical_models;
	vector<string>					hierarchical_model_names;
	vector<vector<pair<int,int>>>	hierarchical_mapping;
	vector<FaceModelParameters>		hierarchical_params;

	//==================== Helpers for face detection and landmark detection validation =========================================

	std::shared_ptr<cl::fd::Detector> 	face_detector;
	std::shared_ptr<cl::fa::Alignment> 	face_alignment;

	string                  face_detector_location;


	// Validate if the detected landmarks are correct using an SVR regressor
	DetectionValidator	landmark_validator; 

	// Indicating if landmark detection succeeded (based on SVR validator)
	bool				detection_success; 

	// Indicating if the tracking has been initialised (for video based tracking)
	bool				tracking_initialised;

	// The actual output of the regressor (-1 is perfect detection 1 is worst detection)
	double				detection_certainty; 

	// Indicator if eye model is there for eye detection
	bool				eye_model;

	// the triangulation per each view (for drawing purposes only)
	vector<cv::Mat_<int> >	triangulations;
	
	//===========================================================================
	// Member variables that retain the state of the tracking (reflecting the state of the lastly tracked (detected) image

	// Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
	cv::Mat_<double>		detected_landmarks;
	
	// The landmark detection likelihoods (combined and per patch expert)
	double				model_likelihood;
	cv::Mat_<double>		landmark_likelihoods;
	
	// Keeping track of how many frames the tracker has failed in so far when tracking in videos
	// This is useful for knowing when to initialise and reinitialise tracking
	int failures_in_a_row;

	// A template of a face that last succeeded with tracking (useful for large motions in video)
	cv::Mat_<uchar> face_template;

	// Useful when resetting or initialising the model closer to a specific location (when multiple faces are present)
	cv::Point_<double> preference_det;

	// A default constructor
	CLNF();

	// Constructor from a model file
    explicit CLNF(const string& model_file);
    explicit CLNF(const FaceModelParameters& param);

    // Copy constructor (makes a deep copy of the detector)
	CLNF(const CLNF& other);

	// Assignment operator for lvalues (makes a deep copy of the detector)
	CLNF & operator= (const CLNF& other);

	// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
	~CLNF(){}

	// Move constructor
	CLNF(const CLNF&& other);

	// Assignment operator for rvalues
	CLNF & operator= (const CLNF&& other);

	// Does the actual work - landmark detection
	bool DetectLandmarks(const cv::Mat_<uchar> &image, FaceModelParameters& params);
	bool init_tracker(const cv::Mat_<uchar> &image, const vector<cv::Point2f>&landmarks, FaceModelParameters& params);
	bool update_tracker(const cv::Mat_<uchar> &image, const vector<cv::Point2f>&landmarks, FaceModelParameters& params);

	// Gets the shape of the current detected landmarks in utils space (given utils calibration)
	// Can only be called after a call to DetectLandmarksInVideo or DetectLandmarksInImage
	cv::Mat_<double> GetShape(double fx, double fy, double cx, double cy) const;

	// A utility bounding box function
	cv::Rect_<double> GetBoundingBox() const;
    cv::Rect_<double> get_bbox(float scale=1.5, float shift_scale = 0) const;

	cv::Mat get_aligned_face(const cv::Mat& src) const;

	// Reset the model (useful if we want to completelly reinitialise, or we want to track another video)
	void Reset();

	// Reset the model, choosing the face nearest (x,y) where x and y are between 0 and 1.
	void Reset(double x, double y);

	// Reading the model in
	void Read(string name);

	// Helper reading function
	void Read_CLNF(string clnf_location);
	
private:

	// the speedup of RLMS using precalculated KDE responses (described in Saragih 2011 RLMS paper)
	map<int, cv::Mat_<float> >		kde_resp_precalc;

	// The model fitting: patch response computation and optimisation steps
    bool Fit(const cv::Mat_<uchar>& intensity_image, const std::vector<int>& window_sizes, const FaceModelParameters& parameters);

	// Mean shift computation that uses precalculated kernel density estimators (the one actually used)
	void NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, map<int, cv::Mat_<float> >& mean_shifts);

	// The actual model optimisation (update step), returns the model likelihood
    double NU_RLMS(cv::Vec6d& final_global, cv::Mat_<double>& final_local, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Vec6d& initial_global, const cv::Mat_<double>& initial_local,
		          const cv::Mat_<double>& base_shape, const cv::Matx22d& sim_img_to_ref, const cv::Matx22f& sim_ref_to_img, int resp_size, int view_idx, bool rigid, int scale, cv::Mat_<double>& landmark_lhoods, const FaceModelParameters& parameters);

	// Generating the weight matrix for the Weighted least squares
	void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters);

	//=======================================================
	// Legacy functions that are not used at the moment
	//=======================================================

	// Mean shift computation	
	void NonVectorisedMeanShift(cv::Mat_<double>& out_mean_shifts, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<double> &dxs, const cv::Mat_<double> &dys, int resp_size, double a, int scale, int view_id);

	// A vectorised version of mean shift (Not actually used)
	void VectorisedMeanShift(cv::Mat_<double>& meanShifts, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<double> &iis, const cv::Mat_<double> &jjs, const cv::Mat_<double> &dxs, const cv::Mat_<double> &dys, const cv::Size patchSize, double sigma, int scale, int view_id);

  };
  //===========================================================================
}
#endif
