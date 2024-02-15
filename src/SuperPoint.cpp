#include "SuperPoint.h"

namespace ORB_SLAM3
{

    const int c1 = 64;
    const int c2 = 64;
    const int c3 = 128;
    const int c4 = 128;
    const int c5 = 256;
    const int d1 = 256;

    SuperPoint::SuperPoint()
        : conv1a(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1)),
          conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

          conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
          conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

          conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
          conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

          conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
          conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

          convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
          convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

          convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
          convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))

    {
        register_module("conv1a", conv1a);
        register_module("conv1b", conv1b);

        register_module("conv2a", conv2a);
        register_module("conv2b", conv2b);

        register_module("conv3a", conv3a);
        register_module("conv3b", conv3b);

        register_module("conv4a", conv4a);
        register_module("conv4b", conv4b);

        register_module("convPa", convPa);
        register_module("convPb", convPb);

        register_module("convDa", convDa);
        register_module("convDb", convDb);
    }

    std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x)
    {

        x = torch::relu(conv1a->forward(x));
        x = torch::relu(conv1b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv2a->forward(x));
        x = torch::relu(conv2b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv3a->forward(x));
        x = torch::relu(conv3b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv4a->forward(x));
        x = torch::relu(conv4b->forward(x));

        auto cPa = torch::relu(convPa->forward(x));
        auto semi = convPb->forward(cPa); // [B, 65, H/8, W/8]

        auto cDa = torch::relu(convDa->forward(x));
        auto desc = convDb->forward(cDa); // [B, d1, H/8, W/8]

        auto dn = torch::norm(desc, 2, 1);
        desc = desc.div(torch::unsqueeze(dn, 1));

        semi = torch::softmax(semi, 1);
        semi = semi.slice(1, 0, 64);
        semi = semi.permute({0, 2, 3, 1}); // [B, H/8, W/8, 64]

        int Hc = semi.size(1);
        int Wc = semi.size(2);
        semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
        semi = semi.permute({0, 1, 3, 2, 4});
        semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]

        std::vector<torch::Tensor> ret;
        ret.push_back(semi);
        ret.push_back(desc);

        return ret;
    }

    SPDetector::SPDetector()
    {
    }

    void SPDetector::build_model()
    {
        model = std::make_shared<SuperPoint>();
        torch::load(model, "./benchmarks/orbslam3_sp_torch/src/original/engines/superpoint_v1.pt");
    }

    void SPDetector::detect(cv::Mat &img, bool cuda)
    {
        auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
        x = x.to(torch::kFloat) / 255;

        bool use_cuda = cuda && torch::cuda::is_available();
        torch::DeviceType device_type;
        if (use_cuda)
            device_type = torch::kCUDA;
        else
            device_type = torch::kCPU;
        torch::Device device(device_type);

        model->to(device);
        x = x.set_requires_grad(false);
        auto out = model->forward(x.to(device));

        mProb = out[0]; // [1, H, W]
        mDesc = out[1]; // [1, 256, H/8, W/8]
    }

    void SPDetector::getKeyPoints(std::vector<cv::KeyPoint> &keypoints, float threshold, int height, int width, int border)
    {
        torch::Tensor prob = mProb.detach().clone();
        simpleNMS(prob, 4);
        prob = prob.squeeze(0);                       // [H, W]
        auto kpts = torch::nonzero(prob > threshold); // [n_keypoints, 2]  (y, x)

        std::vector<cv::KeyPoint> keypoints_nms;
        for (int i = 0; i < kpts.size(0); i++)
        {
            int y = kpts[i][0].item<int>();
            int x = kpts[i][1].item<int>();
            if(x > width-border || x < border || y > height-border || y < border) {
                continue;
            }
            float response = prob[y][x].item<float>();
            keypoints_nms.push_back(cv::KeyPoint(x, y, 1.0, 0.0, response));
        }
        keypoints = keypoints_nms;
    }

    void SPDetector::computeDescriptors(cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, bool cuda)
    {
        int h = mProb.squeeze(0).size(0);
        int w = mProb.squeeze(0).size(1);
        cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)

        for (size_t i = 0; i < keypoints.size(); i++)
        {
            kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
            kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
        }

        auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

        auto grid = torch::zeros({1, 1, fkpts.size(0), 2});                         // [1, 1, n_keypoints, 2]
        grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / w - 1; // x
        grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / h - 1; // y

        if(cuda) {
            grid = grid.to(torch::kCUDA);
        }
        else {
            mDesc = mDesc.to(torch::kCPU);
        }
        auto desc = torch::grid_sampler(mDesc, grid, 0, 0, true); // [1, 256, 1, n_keypoints]
        desc = desc.squeeze(0).squeeze(1);                  // [256, n_keypoints]

        // normalize to 1
        auto dn = torch::norm(desc, 2, 1);
        desc = desc.div(torch::unsqueeze(dn, 1));

        desc = desc.transpose(0, 1).contiguous(); // [n_keypoints, 256]
        desc = desc.to(torch::kCPU);

        cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());

        descriptors = desc_mat.clone();
    }

    void SPDetector::simpleNMS(torch::Tensor &scores, int nms_radius)
    {
        assert(nms_radius >= 0);

        torch::Tensor zeros = torch::zeros_like(scores);
        torch::Tensor maxMask = (scores == torch::max_pool2d(scores, {nms_radius * 2 + 1}, {1}, {nms_radius}));

        for (int i = 0; i < 2; ++i)
        {
            torch::Tensor suppMask = (torch::max_pool2d(maxMask.to(torch::kFloat), {nms_radius * 2 + 1}, {1}, {nms_radius}) > 0);
            torch::Tensor suppScores = torch::where(suppMask, zeros, scores);
            torch::Tensor newMaxMask = (suppScores == torch::max_pool2d(suppScores, {nms_radius * 2 + 1}, {1}, {nms_radius}));
            maxMask = maxMask.logical_or(newMaxMask.logical_and(~suppMask));
        }

        scores = torch::where(maxMask, scores, zeros);
    }
} // namespace ORB_SLAM