#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <vector>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace ORB_SLAM3
{

  struct SuperPoint : torch::nn::Module
  {
    SuperPoint();

    std::vector<torch::Tensor> forward(torch::Tensor x);

    torch::nn::Conv2d conv1a;
    torch::nn::Conv2d conv1b;

    torch::nn::Conv2d conv2a;
    torch::nn::Conv2d conv2b;

    torch::nn::Conv2d conv3a;
    torch::nn::Conv2d conv3b;

    torch::nn::Conv2d conv4a;
    torch::nn::Conv2d conv4b;

    torch::nn::Conv2d convPa;
    torch::nn::Conv2d convPb;

    // descriptor
    torch::nn::Conv2d convDa;
    torch::nn::Conv2d convDb;
  };

  class SPDetector
  {
  public:
    SPDetector();
    void build_model();
    void detect(cv::Mat &image, bool cuda);
    void getKeyPoints(std::vector<cv::KeyPoint> &keypoints, float threshold, int height, int width, int border);
    void computeDescriptors(cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, bool cuda);
    void simpleNMS(torch::Tensor &scores, int nms_radius);

  private:
    std::shared_ptr<SuperPoint> model;
    torch::Tensor mProb;
    torch::Tensor mDesc;
  };

} // ORB_SLAM

#endif