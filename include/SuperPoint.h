#include <opencv2/opencv.hpp>
#include "data_body.h"
#include <NvInferPlugin.h>

#ifndef SuperPoint_H
#define SuperPoint_H

class SuperPoint
{
public:
    SuperPoint();
    ~SuperPoint();
    bool initial_point_model();
    size_t forward(cv::Mat &srcimg, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    template <typename T>
    using _unique_ptr = std::unique_ptr<T, InferDeleter>;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine_ptr;
    std::shared_ptr<nvinfer1::IExecutionContext> _context_ptr;
    point_lm_params _point_lm_params;

private:
    bool build_model();
    void imnormalize(cv::Mat &img, float *blob);
    void* buffers[3];
    float *blob;
    float *scores_output;
    float *descriptors_output;
    cudaStream_t stream_device;
    cudaStream_t stream_input;
    cudaStream_t stream_scores;
    cudaStream_t stream_descriptors;
    bool streamsAreCreated;
};

#endif