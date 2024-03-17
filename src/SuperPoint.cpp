/**
 * Original code from https://github.com/KinglittleQ/SuperPoint_SLAM
 * https://github.com/introlab/rtabmap
 */

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

    SPDetector::SPDetector(const std::string &modelPath, float threshold, bool nms, int minDistance, bool cuda) : threshold_(threshold),
                                                                                                                  nms_(nms),
                                                                                                                  minDistance_(minDistance),
                                                                                                                  detected_(false)
    {
        model_ = std::make_shared<SuperPoint>();
        torch::load(model_, modelPath);

        if (cuda && !torch::cuda::is_available())
        {
            std::cout << "Cuda option is enabled but torch doesn't have cuda support on this platform, using CPU instead." << std::endl;
        }
        cuda_ = cuda && torch::cuda::is_available();
        torch::Device device(cuda_ ? torch::kCUDA : torch::kCPU);
        model_->to(device);
    }

    SPDetector::~SPDetector()
    {
    }

    std::vector<cv::KeyPoint> SPDetector::detect(const cv::Mat &img, const cv::Mat &mask)
    {
        assert(img.type() == CV_8UC1);
        assert(mask.empty() || (mask.type() == CV_8UC1 && img.cols == mask.cols && img.rows == mask.rows));
        detected_ = false;
        if (model_)
        {
            torch::NoGradGuard no_grad_guard;
            auto x = torch::from_blob(img.data, {1, 1, img.rows, img.cols}, torch::kByte);
            x = x.to(torch::kFloat) / 255;

            torch::Device device(cuda_ ? torch::kCUDA : torch::kCPU);
            x = x.set_requires_grad(false);
            auto out = model_->forward(x.to(device));

            auto scores = out[0]; // [1, H, W]
            desc_ = out[1];       // [1, 256, H/8, W/8]

            if (nms_)
            {
                auto options = torch::nn::functional::MaxPool2dFuncOptions(minDistance_ * 2 + 1).stride(1).padding(minDistance_);
                auto options_r1 = torch::nn::functional::MaxPool2dFuncOptions(3).stride(1).padding(1);

                auto zeros = torch::zeros_like(scores);
                auto max_mask = scores == torch::nn::functional::max_pool2d(scores, options);
                auto max_mask_r1 = scores == torch::nn::functional::max_pool2d(scores, options_r1);
                for (size_t i = 0; i < 2; i++)
                {
                    auto supp_mask = torch::nn::functional::max_pool2d(max_mask.to(torch::kF32), options) > 0;
                    auto supp_scores = torch::where(supp_mask, zeros, scores);
                    auto new_max_mask = supp_scores == torch::nn::functional::max_pool2d(supp_scores, options);
                    max_mask = max_mask | (new_max_mask & (~supp_mask) & max_mask_r1);
                }
                prob_ = torch::where(max_mask, scores, zeros).squeeze(0);
            }
            else
            {
                prob_ = scores.squeeze(0);
            }

            auto kpts = (prob_ > threshold_);
            kpts = torch::nonzero(kpts); // [n_keypoints, 2]  (y, x)

            // convert back to cpu if in gpu
            auto kpts_cpu = kpts.to(torch::kCPU);
            auto prob_cpu = prob_.to(torch::kCPU);

            std::vector<cv::KeyPoint> keypoints;
            for (int i = 0; i < kpts_cpu.size(0); i++)
            {
                if (mask.empty() || mask.at<unsigned char>(kpts_cpu[i][0].item<int>(), kpts_cpu[i][1].item<int>()) != 0)
                {
                    float response = prob_cpu[kpts_cpu[i][0]][kpts_cpu[i][1]].item<float>();
                    keypoints.emplace_back(cv::KeyPoint(kpts_cpu[i][1].item<float>(), kpts_cpu[i][0].item<float>(), 8, -1, response));
                }
            }

            detected_ = true;
            return keypoints;
        }
        else
        {
            throw std::runtime_error("No model is loaded!");
            return std::vector<cv::KeyPoint>();
        }
    }

    cv::Mat SPDetector::compute(const std::vector<cv::KeyPoint> &keypoints)
    {
        if (!detected_)
        {
            throw std::runtime_error("SPDetector has been reset before extracting the descriptors! detect() should be called before compute().");
            return cv::Mat();
        }
        if (keypoints.empty())
        {
            return cv::Mat();
        }
        if (model_.get())
        {
            cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)

            // Based on sample_descriptors() of SuperPoint implementation in SuperGlue:
            // https://github.com/magicleap/SuperGluePretrainedNetwork/blob/45a750e5707696da49472f1cad35b0b203325417/models/superpoint.py#L80-L92
            float s = 8;
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y - s / 2 + 0.5;
                kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x - s / 2 + 0.5;
            }

            auto fkpts = torch::from_blob(kpt_mat.data, {(long int)keypoints.size(), 2}, torch::kFloat);

            float w = desc_.size(3); // W/8
            float h = desc_.size(2); // H/8

            torch::Device device(cuda_ ? torch::kCUDA : torch::kCPU);
            auto grid = torch::zeros({1, 1, fkpts.size(0), 2}).to(device);                      // [1, 1, n_keypoints, 2]
            grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / (w * s - s / 2 - 0.5) - 1; // x
            grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / (h * s - s / 2 - 0.5) - 1; // y

            auto desc = torch::grid_sampler(desc_, grid, 0, 0, true); // [1, 256, 1, n_keypoints]

            // normalize to 1
            desc = torch::nn::functional::normalize(desc.reshape({1, desc_.size(1), -1})); //[1, 256, n_keypoints]
            desc = desc.squeeze();                                                         //[256, n_keypoints]
            desc = desc.transpose(0, 1).contiguous();                                      //[n_keypoints, 256]

            if (cuda_)
                desc = desc.to(torch::kCPU);

            cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr<float>());

            return desc_mat.clone();
        }
        else
        {
            throw std::runtime_error("No model is loaded!");
            return cv::Mat();
        }
    }

    void SPDetector::NMS(
        const std::vector<cv::KeyPoint> &ptsIn,
        const cv::Mat &descriptorsIn,
        std::vector<cv::KeyPoint> &ptsOut,
        cv::Mat &descriptorsOut,
        int border, int dist_thresh, int img_width, int img_height)
    {
        std::vector<cv::Point2f> pts_raw;

        for (size_t i = 0; i < ptsIn.size(); i++)
        {
            int u = (int)ptsIn[i].pt.x;
            int v = (int)ptsIn[i].pt.y;

            pts_raw.emplace_back(cv::Point2f(u, v));
        }

        // Grid Value Legend:
        //     255  : Kept.
        //      0   : Empty or suppressed.
        //     100  : To be processed (converted to either kept or suppressed).
        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

        cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);
        cv::Mat dilated_conf = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

        grid.setTo(0);
        inds.setTo(0);
        confidence.setTo(0);

        for (size_t i = 0; i < pts_raw.size(); i++)
        {
            int uu = (int)pts_raw[i].x;
            int vv = (int)pts_raw[i].y;

            grid.at<unsigned char>(vv, uu) = 100;
            inds.at<unsigned short>(vv, uu) = i;

            confidence.at<float>(vv, uu) = ptsIn[i].response;
        }

        cv::dilate(confidence, dilated_conf, cv::Mat());
        cv::Mat peaks = confidence == dilated_conf;

        cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

        for (size_t i = 0; i < pts_raw.size(); i++)
        {
            // account for top left padding
            int uu = (int)pts_raw[i].x + dist_thresh;
            int vv = (int)pts_raw[i].y + dist_thresh;
            float c = confidence.at<float>(vv - dist_thresh, uu - dist_thresh);

            if (grid.at<unsigned char>(vv, uu) == 100) // If not yet suppressed.
            {
                if (peaks.at<unsigned char>(vv - dist_thresh, uu - dist_thresh) == 255)
                {
                    for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
                    {
                        for (int j = -dist_thresh; j < (dist_thresh + 1); j++)
                        {
                            if ((j == 0 && k == 0) || grid.at<unsigned char>(vv + k, uu + j) == 0)
                                continue;

                            if (confidence.at<float>(vv + k - dist_thresh, uu + j - dist_thresh) <= c)
                                grid.at<unsigned char>(vv + k, uu + j) = 0;
                        }
                    }
                    grid.at<unsigned char>(vv, uu) = 255;
                }
                else
                {
                    grid.at<unsigned char>(vv, uu) = 0;
                }
            }
        }

        size_t valid_cnt = 0;
        std::vector<int> select_indice;

        grid = cv::Mat(grid, cv::Rect(dist_thresh, dist_thresh, img_width, img_height));

        for (int v = 0; v < img_height; v++)
        {
            for (int u = 0; u < img_width; u++)
            {
                if (grid.at<unsigned char>(v, u) == 255)
                {
                    int select_ind = (int)inds.at<unsigned short>(v, u);
                    ptsOut.emplace_back(ptsIn[select_ind]);
                    select_indice.emplace_back(select_ind);
                    valid_cnt++;
                }
            }
        }

        if (!descriptorsIn.empty())
        {
            assert(descriptorsIn.rows == (int)ptsIn.size());
            descriptorsOut.create(select_indice.size(), 256, CV_32F);

            for (size_t i = 0; i < select_indice.size(); i++)
            {
                descriptorsIn.row(select_indice[i]).copyTo(descriptorsOut.row(i));
            }
        }
    }

}
