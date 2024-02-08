/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <Parameters.h>
#include <SLAMBenchAPI.h>
#include <System.h>
#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/DepthSensor.h>
#include <io/sensor/GroundTruthSensor.h>
#include <io/sensor/IMUSensor.h>
#include <chrono>
#include <Eigen/Eigen>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#if (CV_MAJOR_VERSION > 3)
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/calib3d/calib3d_c.h>
#else
#include <opencv2/core/core.hpp>
#endif
// access to slam objects
static cv::Mat pose;
static cv::Mat frameCV;
static sb_uint2 inputSize;

static cv::Mat result_tracking;
static cv::Mat *imD = nullptr;
static cv::Mat *imRGB = nullptr;

cv::Mat* img_one;
cv::Mat* img_two;

enum orbslam_input_mode {mono,stereo,rgbd,monoimu,stereoimu,automatic};

static orbslam_input_mode input_mode;
static const orbslam_input_mode default_input_mode = orbslam_input_mode::automatic;

static std::string settings_file;
static std::string vocabulary_file;


static slambench::outputs::Output *pose_output;
static slambench::outputs::Output *pointcloud_output;

static slambench::outputs::Output *frame1_output;
static slambench::outputs::Output *frame2_output;

static slambench::TimeStamp last_frame_timestamp;

static const std::string default_settings_file = "";
static const std::string default_vocabulary_file = "./benchmarks/orbslam3_sp_trt/src/original/Vocabulary/ORBvoc.txt";

ORB_SLAM3::System* SLAM;

// ===========================================================
// SLAMBench Sensors
// ===========================================================

static slambench::io::IMUSensor *IMU_sensor;
static slambench::io::DepthSensor *depth_sensor;
static slambench::io::CameraSensor *rgb_sensor;

static slambench::io::CameraSensor *grey_sensor_one = nullptr;
static slambench::io::CameraSensor *grey_sensor_two = nullptr;

std::vector<ORB_SLAM3::IMU::Point> imupoints;
// ===========================================================
// Variable parameters
// ===========================================================

static int max_features;
static const int default_max_features=1000;
static int pyramid_levels;
static const int default_pyramid_levels=8;
static float scale_factor;
static const float default_scale_factor=1.2;
static int initial_fast_threshold;
static const int default_initial_fast_threshold=20;
static int second_fast_threshold;
static const int default_second_fast_threshold=7;



static int camera_fps;
static const int default_camera_fps=40;

static int frame_no = 0;
static int start_frame;
static const int default_start_frame=0;

static float depth_threshold;
static const float default_depth_threshold=40;

enum copyFrom {SB_TO_CV, CV_TO_SB};
bool sb_get_tracked()  {
    return (SLAM->GetTrackingState() == ORB_SLAM3::Tracking::OK);
}
bool sb_get_initialized()  {
    return !(SLAM->GetTrackingState() == ORB_SLAM3::Tracking::SYSTEM_NOT_READY ||
             SLAM->GetTrackingState() == ORB_SLAM3::Tracking::NO_IMAGES_YET ||
             SLAM->GetTrackingState() == ORB_SLAM3::Tracking::NOT_INITIALIZED ||
             SLAM->GetTrackingState() == ORB_SLAM3::Tracking::RECENTLY_LOST);
}

template<typename TYPE>
void copyIntrinsics(slambench::io::CameraSensor* sensor, cv::Mat& camera_parameters)
{
    static_assert(std::is_same<TYPE, float>() || std::is_same<TYPE, double>(), "Use double for CV_64F and float for CV_32F!");

    camera_parameters.at<TYPE>(0,0) = sensor->Intrinsics[0]*sensor->Width;
    camera_parameters.at<TYPE>(1,1) = sensor->Intrinsics[1]*sensor->Height;
    camera_parameters.at<TYPE>(0,2) = sensor->Intrinsics[2]*sensor->Width;
    camera_parameters.at<TYPE>(1,2) = sensor->Intrinsics[3]*sensor->Height;
}

template<typename TYPE>
void copyDistortion(slambench::io::CameraSensor* sensor, cv::Mat& camera_distortion)
{
    static_assert(std::is_same<TYPE, float>() || std::is_same<TYPE, double>(), "Use double for CV_64F and float for CV_32F!");

    camera_distortion.at<TYPE>(0) = sensor->Distortion[0];
    camera_distortion.at<TYPE>(1) = sensor->Distortion[1];
    camera_distortion.at<TYPE>(2) = sensor->Distortion[2];
    camera_distortion.at<TYPE>(3) = sensor->Distortion[3];
    camera_distortion.at<TYPE>(4) = sensor->Distortion[4];
}

template<typename TYPE>
void copyPose(Eigen::Matrix<TYPE,4,4> &sb_pose, cv::Mat& sensor_pose, copyFrom copy_direction = copyFrom::CV_TO_SB)
{
    static_assert(std::is_same<TYPE, float>() || std::is_same<TYPE, double>(), "Use double for CV_64F and float for CV_32F!");
    if(std::is_same<TYPE, float>())
        assert(sensor_pose.type() == CV_32F && "Use float for CV_32F!");
    else if(std::is_same<TYPE, double>())
        assert(sensor_pose.type() == CV_64F && "Use double for CV_64F!");

    for(size_t i = 0; i < 4; i++) {
        for(size_t j = 0; j < 4; j++) {
            if(copy_direction == copyFrom::CV_TO_SB)
                sb_pose(i,j) = sensor_pose.at<TYPE>(i,j);
            else
                sensor_pose.at<TYPE>(i,j) = sb_pose(i,j);
        }
    }

}

// ===========================================================
// PERSONALIZED DATATYPE FOR ORBSLAM PARAMETERS
// ===========================================================
template<> inline const std::string TypedParameter<orbslam_input_mode>::getValue(const void * ptr) const {
    switch (*((orbslam_input_mode*) ptr))  {
        case orbslam_input_mode::mono : return "mono";
        case orbslam_input_mode::stereo : return "stereo";
        case orbslam_input_mode::rgbd : return "rgbd";
        case orbslam_input_mode::monoimu : return "monoimu";
        case orbslam_input_mode::stereoimu : return "stereoimu";
        case orbslam_input_mode::automatic : return "auto";
    }
    return "error";
};

template<> inline void  TypedParameter<orbslam_input_mode>::copyValue(orbslam_input_mode *to , orbslam_input_mode const *from) {
    *to = *from;
};

template<> inline void  TypedParameter<orbslam_input_mode>::setValue(const char* optarg)  {

    if (std::string(optarg) == "auto")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::automatic;}
    else if (std::string(optarg) == "mono")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::mono;}
    else if (std::string(optarg) == "stereo")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::stereo;}
    else if (std::string(optarg) == "rgbd")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::rgbd;}
    else if (std::string(optarg) == "monoimu")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::monoimu;}
    else if (std::string(optarg) == "stereoimu")
    {*((orbslam_input_mode*)ptr_) = orbslam_input_mode::stereoimu;}
    else
    {throw std::logic_error("The argument you gave for ORBSLAM Mode is incorrect, only 'auto', 'mono', 'stereo', 'rgbd', 'monoimu' or 'stereoimu' are valid.");}
};


// ===========================================================
// Rectification for Stereo mode
// ===========================================================
cv::Mat M1l,M2l,M1r,M2r;

// ===========================================================
// SLAMBENCH API
// ===========================================================


bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {

    slam_settings->addParameter(TypedParameter<orbslam_input_mode>("m", "mode",     "select input mode (auto,mono,stereo,rgbd)",    &input_mode, &default_input_mode));
    slam_settings->addParameter(TypedParameter<std::string>("", "settings",     "Path to the setting file",    &settings_file, &default_settings_file));
    slam_settings->addParameter(TypedParameter<std::string>("voc", "vocabulary",     "Path to the vocabulary file",    &vocabulary_file, &default_vocabulary_file));

    // algo parameters
    slam_settings->addParameter(TypedParameter<int>("mf", "max-features",     "Maximum number of features",    &max_features, &default_max_features));
    slam_settings->addParameter(TypedParameter<int>("sl", "scale-levels",     "Number of levels in image pyramid",    &pyramid_levels, &default_pyramid_levels));
    slam_settings->addParameter(TypedParameter<float>("sf", "scale-factor",     "Scale between levels in image pyramid",    &scale_factor, &default_scale_factor));
    slam_settings->addParameter(TypedParameter<int>("ift", "initial-fast-threshold",     "Initial threshold of FAST algorithm (high)",    &initial_fast_threshold,
                                                    &default_initial_fast_threshold));
    slam_settings->addParameter(TypedParameter<int>("sft", "second-fast-threshold",     "Second threshold of FAST algorithm (low)",    &second_fast_threshold,
                                                    &default_second_fast_threshold));

    // camera parameters
    slam_settings->addParameter(TypedParameter<int>("fps", "camera-fps",     "Camera frame rate",    &camera_fps,  &default_camera_fps));
    slam_settings->addParameter(TypedParameter<float>("dt", "depth-threshold",     "Depth threshold (close/far points)",    &depth_threshold,  &default_depth_threshold));
    slam_settings->addParameter(TypedParameter<int>("", "start-frame", "first frame to compute", &start_frame, &default_start_frame));

    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)  {


    //=========================================================================
    // We collect sensors
    //=========================================================================

    slambench::io::CameraSensorFinder sensor_finder;
    rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
    depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});
    IMU_sensor = (slambench::io::IMUSensor*)slam_settings->get_sensors().GetSensor(slambench::io::IMUSensor::kIMUType);
    auto grey_sensors = sensor_finder.Find(slam_settings->get_sensors(), {{"camera_type", "grey"}});

    if(grey_sensors.size() == 2) {
        grey_sensor_one = grey_sensors.at(0);
        grey_sensor_two = grey_sensors.at(1);
    }

    if(input_mode == orbslam_input_mode::automatic) {
        if (rgb_sensor || grey_sensor_one)
        {
            input_mode = orbslam_input_mode::mono;
            if(IMU_sensor)
                input_mode = orbslam_input_mode::monoimu;
        }
        if (rgb_sensor && depth_sensor)
            input_mode = orbslam_input_mode::rgbd;
        if (grey_sensor_one && grey_sensor_two)
        {
            input_mode = orbslam_input_mode::stereo;
            if(IMU_sensor)
                input_mode = orbslam_input_mode::stereoimu;
        }
    }

    if(input_mode == orbslam_input_mode::rgbd) {

        if ((rgb_sensor == nullptr) or (depth_sensor == nullptr)) {
            std::cerr << "Invalid sensors found, RGB or Depth not found." << std::endl;
            std::cerr << "RGB   = " << (rgb_sensor == nullptr) << std::endl;
            std::cerr << "DEPTH = " << (depth_sensor == nullptr) << std::endl;
            return false;
        }

        if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
            std::cerr << "RGB data is in wrong format" << std::endl;
            return false;
        }
        if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
            std::cerr << "Depth data is in wrong format" << std::endl;
            return false;
        }
        if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
            std::cerr << "RGB data is in wrong format pixel" << std::endl;
            return false;
        }
        if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
            std::cerr << "Depth data is in wrong pixel format" << std::endl;
            return false;
        }

        if(rgb_sensor->Width != depth_sensor->Width || rgb_sensor->Height != depth_sensor->Height) {
            std::cerr << "Sensor size mismatch" << std::endl;
            return false;
        }

        cv::Mat camera_parameters = cv::Mat::eye(3,3,CV_32F);
        copyIntrinsics<float>(rgb_sensor, camera_parameters);

        cv::Mat camera_distortion(1,5,CV_32F);
        copyDistortion<float>(rgb_sensor, camera_distortion);

        imRGB = new cv::Mat (rgb_sensor->Height, rgb_sensor->Width,CV_8UC3);
        imD   = new cv::Mat (depth_sensor->Height, depth_sensor->Width,CV_16UC1);
        inputSize   = make_sb_uint2(rgb_sensor->Width, rgb_sensor->Height);
//       Fixme: depth factor is not in disparity params.
        float DepthMapFactor = depth_sensor->DisparityParams[0];
        SLAM = new ORB_SLAM3::System(vocabulary_file,settings_file,ORB_SLAM3::System::RGBD,false);

        if(settings_file.empty()) {
            SLAM->mpTracker->ConfigureCamera(camera_parameters,cv::Mat(),  camera_distortion, camera_fps, DepthMapFactor, 40, depth_threshold);
            SLAM->mpTracker->ConfigureAlgorithm(max_features, pyramid_levels, scale_factor, initial_fast_threshold, second_fast_threshold);
        }

    } else if (input_mode == orbslam_input_mode::mono || input_mode == orbslam_input_mode::monoimu) {

        if (rgb_sensor == nullptr and grey_sensor_one ==nullptr) {
            std::cerr << "Invalid sensors found, RGB or Grey sensor not found." << std::endl;
            return false;
        }

        if(rgb_sensor != nullptr and rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
            std::cerr << "RGB data is in wrong format" << std::endl;
            return false;
        }

        if(rgb_sensor != nullptr and rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
            std::cerr << "RGB data is in wrong format pixel" << std::endl;
            return false;
        }

        cv::Mat camera_parameters = cv::Mat::eye(3,3,CV_32F);
        cv::Mat camera_distortion(1,5, CV_32F);

        if (rgb_sensor) {
            copyIntrinsics<float>(rgb_sensor, camera_parameters);
            copyDistortion<float>(rgb_sensor, camera_distortion);
            imRGB       = new cv::Mat(rgb_sensor->Height, rgb_sensor->Width,CV_8UC3);
            inputSize   = make_sb_uint2(rgb_sensor->Width, rgb_sensor->Height);
        } else {
            assert(grey_sensor_one);
            copyIntrinsics<float>(grey_sensor_one, camera_parameters);
            copyDistortion<float>(grey_sensor_one, camera_distortion);

            img_one     = new cv::Mat(grey_sensor_one->Height, grey_sensor_one->Width,CV_8UC1);
            inputSize   = make_sb_uint2(grey_sensor_one->Width, grey_sensor_one->Height);
        }

        if(input_mode == orbslam_input_mode::monoimu && IMU_sensor != nullptr)
            SLAM = new ORB_SLAM3::System(vocabulary_file, settings_file,ORB_SLAM3::System::IMU_MONOCULAR,false);
        else
        {
            input_mode = orbslam_input_mode::mono;
            SLAM = new ORB_SLAM3::System(vocabulary_file, settings_file,ORB_SLAM3::System::MONOCULAR,false);
        }

        if(settings_file.empty()) {
            SLAM->mpTracker->ConfigureCamera(camera_parameters, cv::Mat(), camera_distortion, camera_fps, 1, 40, depth_threshold);
            SLAM->mpTracker->ConfigureAlgorithm(max_features, pyramid_levels, scale_factor, initial_fast_threshold, second_fast_threshold);
            if(input_mode == orbslam_input_mode::monoimu)
            {
                cv::Mat Tbc = cv::Mat::zeros(4,4,CV_32F);
                if(rgb_sensor)
                    copyPose<float>(rgb_sensor->Pose, Tbc, copyFrom::SB_TO_CV);
                else
                    copyPose<float>(rgb_sensor->Pose, Tbc, copyFrom::SB_TO_CV);

                SLAM->mpTracker->ConfigureIMU(Tbc,
                                              IMU_sensor->Rate,
                                              IMU_sensor->GyroscopeNoiseDensity,
                                              IMU_sensor->AcceleratorNoiseDensity,
                                              IMU_sensor->GyroscopeBiasDiffusion,
                                              IMU_sensor->AcceleratorBiasDiffusion);
            }

        }
    } else if (input_mode == orbslam_input_mode::stereo || input_mode == orbslam_input_mode::stereoimu) {

        if(grey_sensor_one == nullptr or grey_sensor_two == nullptr) {
            std::cerr << "Invalid sensors found, Grey Stereo not found." << std::endl;
            return false;
        }
        if(grey_sensor_one->DistortionType != grey_sensor_two->DistortionType) {
            std::cerr << "Stereo cameras with different distortion types not supported!" << std::endl;
            return false;
        }

        cv::Mat K_l, K_r, D_l, D_r;

        K_l = cv::Mat::eye(3, 3,CV_64F);
        copyIntrinsics<double>(grey_sensor_one, K_l);

        K_r = cv::Mat::eye(3, 3,CV_64F);
        copyIntrinsics<double>(grey_sensor_two, K_r);

        int num_dist_params = 4; // Equidistant or Kannala-Brandt
        if(grey_sensor_one->DistortionType == slambench::io::CameraSensor::RadialTangential)
            num_dist_params = 5;
        D_l = cv::Mat::zeros(1, num_dist_params,CV_64F);
        copyDistortion<double>(grey_sensor_one, D_l);

        D_r = cv::Mat::zeros(1, num_dist_params,CV_64F);
        copyDistortion<double>(grey_sensor_two, D_r);

        std::vector<cv::Mat> vK, vD, vTBS;
        std::vector<cv::Size> vSz;

        // here we read T_BS, K, D and images size for each camera
        cv::Mat T_BS_l(4,4, CV_64F);
        cv::Mat T_BS_r(4,4, CV_64F);
        Eigen::Matrix4d gray_pose_one = grey_sensor_one->Pose.cast<double>();
        Eigen::Matrix4d gray_pose_two = grey_sensor_two->Pose.cast<double>();

        copyPose<double>(gray_pose_one, T_BS_l, copyFrom::SB_TO_CV);
        copyPose<double>(gray_pose_two, T_BS_r, copyFrom::SB_TO_CV);

        cv::Mat P_l, P_r, R_l, R_r;
        cv::Mat R1,R2,P1,P2,Q;
        cv::Mat Tr = (T_BS_r).inv() * (T_BS_l);
        cv::Mat R, T;

        Tr.colRange(0,3).rowRange(0,3).copyTo(R);
        Tr.col(3).rowRange(0,3).copyTo(T);

        int rows_l = grey_sensor_one->Height;
        int cols_l = grey_sensor_one->Width;
        int rows_r = grey_sensor_two->Height;
        int cols_r = grey_sensor_two->Width;

        // note that order of cameras matter (left camera) should be the first one.
        if(grey_sensor_one->DistortionType == slambench::io::CameraSensor::RadialTangential)
            cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(cols_l,rows_l), R, T, R_l, R_r, P_l, P_r, Q, CV_CALIB_ZERO_DISPARITY,0);
        else
            cv::fisheye::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(cols_l,rows_l), R, T, R_l, R_r, P_l, P_r, Q, CV_CALIB_ZERO_DISPARITY);

        double bf = std::abs(P_r.at<double>(0,3) - P_l.at<double>(0,3));

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = P_l.at<double>(0,0);
        K.at<float>(1,1) = P_l.at<double>(1,1);
        K.at<float>(0,2) = P_l.at<double>(0,2);
        K.at<float>(1,2) = P_l.at<double>(1,2);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = 0.0;
        DistCoef.at<float>(1) = 0.0;
        DistCoef.at<float>(2) = 0.0;
        DistCoef.at<float>(3) = 0.0;

        img_one = new cv::Mat(grey_sensor_one->Height, grey_sensor_one->Width,CV_8UC1);
        img_two = new cv::Mat(grey_sensor_two->Height, grey_sensor_two->Width,CV_8UC1);
        inputSize = make_sb_uint2(grey_sensor_one->Width, grey_sensor_one->Height);

        if(input_mode == orbslam_input_mode::stereoimu && IMU_sensor != nullptr)
            SLAM = new ORB_SLAM3::System(vocabulary_file, settings_file,ORB_SLAM3::System::IMU_STEREO,false);
        else
        {
            input_mode = orbslam_input_mode::stereo;
            SLAM = new ORB_SLAM3::System(vocabulary_file, settings_file,ORB_SLAM3::System::STEREO,false);
        }

        if(settings_file.empty()) {
            SLAM->mpTracker->ConfigureCamera(K, cv::Mat(), DistCoef, camera_fps, 1, bf , depth_threshold);
            SLAM->mpTracker->ConfigureAlgorithm(max_features,pyramid_levels,scale_factor,initial_fast_threshold,second_fast_threshold);
            if(input_mode == orbslam_input_mode::stereoimu)
            {
                cv::Mat Tbc(4,4, CV_32F);

                Eigen::Matrix4f imu_pose = IMU_sensor->Pose;
                copyPose<float>(imu_pose, Tbc, copyFrom::SB_TO_CV);

                SLAM->mpTracker->ConfigureIMU(Tbc,
                                              IMU_sensor->Rate,
                                              IMU_sensor->GyroscopeNoiseDensity,
                                              IMU_sensor->AcceleratorNoiseDensity,
                                              IMU_sensor->GyroscopeBiasDiffusion,
                                              IMU_sensor->AcceleratorBiasDiffusion);
            }

        }
        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
           rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            std::cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
            exit(1);
        }

        if(grey_sensor_one->DistortionType == slambench::io::CameraSensor::RadialTangential) {
            cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
            cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
        }
        else {
            cv::fisheye::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
            cv::fisheye::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
        }

    }  else if (input_mode == orbslam_input_mode::automatic) {
        std::cout << "No valid sensor found." << std::endl;
        exit(1);
    } else {
        std::cout << "Invalid input mode '" <<   input_mode << "'" << std::endl;
        exit(1);
    }
    SLAM->mpTracker->PrintConfig();

    pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
    slam_settings->GetOutputManager().RegisterOutput(pose_output);

    pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_POINTCLOUD, true);
    pointcloud_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);

    frame1_output = new slambench::outputs::Output("Input Frame", slambench::values::VT_FRAME);
    frame1_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(frame1_output);

    frame2_output = new slambench::outputs::Output("Tracking frame", slambench::values::VT_FRAME);
    frame2_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(frame2_output);

    return true;
}

bool depth_ready = false, rgb_ready = false, grey_one_ready = false, grey_two_ready = false, imu_ready = false;

bool performTracking()
{
    if (input_mode == orbslam_input_mode::rgbd) {
        pose = SLAM->TrackRGBD(*imRGB,*imD,last_frame_timestamp.ToS());
        std::cout<<"Pose in SLAMBench:"<<pose<<std::endl;
    } else if (input_mode == orbslam_input_mode::mono || input_mode == orbslam_input_mode::monoimu) {
        if (rgb_ready)
            pose = SLAM->TrackMonocular(*imRGB,last_frame_timestamp.ToS(), imupoints);
        else if (grey_one_ready)
            pose = SLAM->TrackMonocular(*img_one,last_frame_timestamp.ToS(), imupoints);
    } else if(input_mode == orbslam_input_mode::stereo || input_mode == orbslam_input_mode::stereoimu) {
        cv::Mat imLeftRect, imRightRect;
        cv::remap(*img_one,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(*img_two,imRightRect,M1r,M2r,cv::INTER_LINEAR);
        pose = SLAM->TrackStereo(imLeftRect,imRightRect,last_frame_timestamp.ToS(), imupoints);
    } else {
        std::cout << "Unsupported case." << std::endl;
        return false;
    }
    frame_no++;
    imupoints.clear();
    imu_ready = false, depth_ready = false, rgb_ready = false, grey_one_ready = false, grey_two_ready = false;
    return true;
}
bool is_cam_frame;
bool switched_dataset = false;
bool sb_update_frame (SLAMBenchLibraryHelper *slam_settings , slambench::io::SLAMFrame* s) {
    assert(s != nullptr);
    is_cam_frame = true;
    if(s->FrameSensor->GetType() == slambench::io::GroundTruthSensor::kGroundTruthTrajectoryType and !sb_get_tracked()) {
        cv::Mat matrix(4,4,CV_32F);
        memcpy(matrix.data, s->GetData(), s->GetSize());
        SLAM->mpTracker->mpMapDrawer->SetCurrentCameraPose(matrix);
        s->FreeData();
    }
        //  Prevent last_frame_timestamp to be updated with IMU sensor timestamp
    else if(s->FrameSensor == depth_sensor and imD) {
        memcpy(imD->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        depth_ready = true;
        s->FreeData();
    } else if(s->FrameSensor == rgb_sensor and imRGB) {
        memcpy(imRGB->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        rgb_ready = true;
        s->FreeData();
    } else if(s->FrameSensor == grey_sensor_one and img_one) {
        memcpy(img_one->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        grey_one_ready = true;
        s->FreeData();
    } else if(s->FrameSensor == grey_sensor_two and img_two) {
        memcpy(img_two->data, s->GetData(), s->GetSize());
        last_frame_timestamp = s->Timestamp;
        grey_two_ready = true;
        s->FreeData();
    }
    else if(s->FrameSensor == IMU_sensor && (input_mode == orbslam_input_mode::stereoimu || input_mode == orbslam_input_mode::monoimu)) {
        float* frame_data = (float*)s->GetData();
//        accelerometer first then gyro
//        FIXME: only add measurements before current camera frame?
        imupoints.push_back(ORB_SLAM3::IMU::Point(frame_data[3], frame_data[4], frame_data[5],
                                                  frame_data[0],frame_data[1],frame_data[2],
                                                  s->Timestamp.ToS()));
        imu_ready = true;
        is_cam_frame = false;
        s->FreeData();
    }
    bool sensors_ready = (input_mode == orbslam_input_mode::rgbd and depth_ready and rgb_ready) or
                         (input_mode == orbslam_input_mode::mono and rgb_ready) or
                         (input_mode == orbslam_input_mode::mono and grey_one_ready) or
                         (input_mode == orbslam_input_mode::monoimu and imu_ready and rgb_ready) or
                         (input_mode == orbslam_input_mode::monoimu and imu_ready and grey_one_ready) or
                         (input_mode == orbslam_input_mode::stereo and grey_one_ready and grey_two_ready) or
                         (input_mode == orbslam_input_mode::stereoimu and grey_one_ready and grey_two_ready and imu_ready);

//  Continue sending in frames if not yet initialized or start frame not reached yet
    if((sensors_ready && !sb_get_initialized()) || frame_no < start_frame)
    {
        cout<<"Perform tracking from sb_update_frame"<<std::endl;
        performTracking();
        return false;
    }

    return sensors_ready;
}

bool sb_process_once (SLAMBenchLibraryHelper *slam_settings)  {
    cout<<"Perform tracking from sb_process_once"<<std::endl;
    if(!performTracking())
        return false;

    SLAM->mpFrameDrawer->setState(SLAM->mpTracker->mLastProcessedState);
    if(!sb_get_tracked())
        std::cerr<<"Tracking failed!"<<std::endl;
//        SLAM->Relocalize();
    return true;
}

// Report last valid pose if tracking is lost
Eigen::Matrix4f last_valid_pose=Eigen::Matrix4f::Identity();

void getAllPoses(std::vector<Eigen::Matrix4f> &sb_poses) {
    auto cv_poses = SLAM->getAllPoses();
    sb_poses.resize(cv_poses.size() - 2);
    for(size_t i = 0; i < sb_poses.size(); i++) //wtf
        copyPose<float>(sb_poses[i], reinterpret_cast<cv::Mat &>(cv_poses[i]), copyFrom::CV_TO_SB);
}

bool sb_get_pose (Eigen::Matrix4f* mat)  {
    for(int j=0; j<4;j++) {
        for(int i=0; i<4;i++) {
            (*mat)(j,i)= pose.at<float>(j,i);
        }
    }
    return true;
}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output) {
    (void)lib;
    auto ts = *latest_output;

    if(pose_output->IsActive()) {
        // Get the current pose as an eigen matrix
        Eigen::Matrix4f matrix;
        if(sb_get_tracked())
        {
            pose=SLAM->mpTracker->getPose();
            sb_get_pose(&matrix);
            last_valid_pose = matrix;
        }
        else
        {
            matrix = last_valid_pose;
        }
        std::lock_guard<FastLock> lock(lib->GetOutputManager().GetLock());
        pose_output->AddPoint(ts, new slambench::values::PoseValue(matrix));
    }

    if(pointcloud_output->IsActive()) {
        auto point_cloud = new slambench::values::PointCloudValue();
        auto vpMPs = SLAM->mpMapDrawer->mpAtlas->GetReferenceMapPoints();
        for(auto & vpMP : vpMPs) {
            if(vpMP->isBad())
                continue;
            cv::Mat pos = vpMP->GetWorldPos();
            point_cloud->AddPoint(slambench::values::Point3DF(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2)));
        }

        // Take lock only after generating the map
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pointcloud_output->AddPoint(ts, point_cloud);
    }

    if(frame1_output->IsActive()) {
        std::lock_guard<FastLock> lock(lib->GetOutputManager().GetLock());
        if (imRGB)
            frame1_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, (void*)(imRGB->data)));
        else if (img_one)
            frame1_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::G_I_8, (void*)(img_one->data)));
    }

    if(frame2_output->IsActive()) {
        frameCV = SLAM->mpFrameDrawer->DrawFrame();
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        frame2_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888,  (void*)(&frameCV.at<char>(0,0))));
    }

    return true;
}


bool sb_clean_slam_system() {
    delete SLAM;
//    SLAM->Shutdown(); FIXME: this hangs
    return true;
}

bool sb_relocalize(SLAMBenchLibraryHelper *lib)
{
    SLAM->ChangeDataset();
//    return SLAM->Relocalize();
    return sb_get_tracked();
}
