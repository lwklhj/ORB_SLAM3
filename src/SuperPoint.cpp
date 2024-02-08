#include <NvInfer.h>
#include "cuda_runtime_api.h"
#include <math.h>

#include "SuperPoint.h"

#include "logging.h"
#include <memory>
#include <fstream>
#include "tools.h"

#include <algorithm>
#include <chrono>

static Logger gLogger;

bool SuperPoint::build_model()
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
    {
        return false;
    }

    char *model_deser_buffer{nullptr};
    const std::string engine_file_path(_point_lm_params.point_weight_file);
    std::ifstream ifs;
    int ser_length;
    ifs.open(engine_file_path.c_str(), std::ios::in | std::ios::binary);
    if (ifs.is_open())
    {
        ifs.seekg(0, std::ios::end);
        ser_length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        model_deser_buffer = new char[ser_length];
        ifs.read(model_deser_buffer, ser_length);
        ifs.close();
    }
    else
    {
        return false;
    }

    _engine_ptr = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_deser_buffer, ser_length), InferDeleter());
    if (!_engine_ptr)
    {
        return false;
    }
    else
    {
        std::cout << "load engine successed!" << std::endl;
    }
    delete[] model_deser_buffer;

    _context_ptr = std::shared_ptr<nvinfer1::IExecutionContext>(_engine_ptr->createExecutionContext(), InferDeleter());
    if (!_context_ptr)
    {
        return false;
    }
    return true;
}

bool SuperPoint::initial_point_model()
{
    bool ret = build_model();
    if (!ret)
    {
        std::cout << "build point model is failed!" << std::endl;
    }
    return ret;
}

size_t SuperPoint::forward(cv::Mat &srcimg, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // Input size
    int dummy_input_size = srcimg.rows * srcimg.cols;
    // Score size
    int scores_size = 1 * srcimg.rows * srcimg.cols;
    // Descriptors size
    int desc_fea_c = 256;
    int desc_fea_h = srcimg.rows / 8;
    int desc_fea_w = srcimg.cols / 8;
    int descriptors_size = 1 * desc_fea_c * desc_fea_h * desc_fea_w;

    // Normalise image and write to pinned memory
    float *blob(nullptr);
    cudaHostAlloc((void **)&blob, dummy_input_size * sizeof(float), cudaHostAllocDefault);
    imnormalize(srcimg, blob);
    if (blob == nullptr)
    {
        std::cout << "imnormalize error! " << std::endl;
        // dp.status_code = 0;
        return -1;
    }

    // Create output arrays
    float *scores_output;
    cudaHostAlloc((void **)&scores_output, scores_size * sizeof(float), cudaHostAllocDefault);
    float *descriptors_output;
    cudaHostAlloc((void **)&descriptors_output, descriptors_size * sizeof(float), cudaHostAllocDefault);

    // Get indexes
    const int dummy_inputIndex = _engine_ptr->getBindingIndex(_point_lm_params.input_names[0].c_str());
    const int scores_outputIndex = _engine_ptr->getBindingIndex(_point_lm_params.output_names[0].c_str());
    const int descriptors_outputIndex = _engine_ptr->getBindingIndex(_point_lm_params.output_names[1].c_str());
    assert(_engine_ptr->getBindingDataType(dummy_inputIndex) == nvinfer1::DataType::kHALF);
    assert(_engine_ptr->getBindingDataType(scores_outputIndex) == nvinfer1::DataType::kFLOAT);
    assert(_engine_ptr->getBindingDataType(descriptors_outputIndex) == nvinfer1::DataType::kFLOAT);

    // Set input shape
    _context_ptr->setInputShape(_point_lm_params.input_names[0].c_str(), nvinfer1::Dims4(1, 1, srcimg.rows, srcimg.cols));

    // Allocate device memory
    void *buffers[3];
    cudaMalloc(&buffers[dummy_inputIndex], dummy_input_size * sizeof(float));
    cudaMalloc(&buffers[scores_outputIndex], scores_size * sizeof(float));
    cudaMalloc(&buffers[descriptors_outputIndex], descriptors_size * sizeof(float));

    // Create cuda streams
    cudaStream_t stream[4];
    for (int i = 0; i < 4; i++)
    {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }

    // Copy input to device
    cudaMemcpyAsync(buffers[dummy_inputIndex], blob, dummy_input_size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    cudaDeviceSynchronize();

    // For enqueueV3
    _context_ptr->setInputTensorAddress(_point_lm_params.input_names[0].c_str(), buffers[dummy_inputIndex]);
    _context_ptr->setTensorAddress(_point_lm_params.output_names[0].c_str(), buffers[scores_outputIndex]);
    _context_ptr->setTensorAddress(_point_lm_params.output_names[1].c_str(), buffers[descriptors_outputIndex]);

    // Start
    bool status = _context_ptr->enqueueV3(stream[1]);
    cudaDeviceSynchronize();

    // delete[] blob;
    cudaFreeHost(blob);
    // blob = nullptr;

    if (!status)
    {
        std::cout << "execute ifer error! " << std::endl;
        // dp.status_code = 0;
        return -1;
    }
    cudaMemcpyAsync(scores_output, buffers[scores_outputIndex], scores_size * sizeof(float), cudaMemcpyDeviceToHost, stream[2]);
    cudaMemcpyAsync(descriptors_output, buffers[descriptors_outputIndex], descriptors_size * sizeof(float), cudaMemcpyDeviceToHost, stream[3]);
    cudaDeviceSynchronize();

    cudaFreeAsync(buffers[dummy_inputIndex], stream[0]);
    cudaFreeAsync(buffers[scores_outputIndex], stream[1]);
    cudaFreeAsync(buffers[descriptors_outputIndex], stream[2]);

    // Process keypoints
    int scores_h = srcimg.rows;
    int scores_w = srcimg.cols;
    int border = _point_lm_params.border;
    // keypoints.reserve(scores_h * scores_w);
    for (int y = border; y < (scores_h - border); y++)
    {
        for (int x = border; x < (scores_w - border); x++)
        {
            float score = scores_output[y * scores_w + x];
            if (score > _point_lm_params.scores_thresh)
            {
                keypoints.push_back(cv::KeyPoint(cv::Point(x, y), 1.0, 0.0, score));
            }
        }
    }
    // keypoints.shrink_to_fit();
    int vdt_size = keypoints.size();

    float *desc_channel_sum_sqrts = new float[desc_fea_h * desc_fea_w];
    for (int dfh = 0; dfh < desc_fea_h; dfh++)
    {
        for (int dfw = 0; dfw < desc_fea_w; dfw++)
        {
            float desc_channel_sum_temp = 0.f;
            for (int dfc = 0; dfc < desc_fea_c; dfc++)
            {
                desc_channel_sum_temp += descriptors_output[dfc * desc_fea_w * desc_fea_h + dfh * desc_fea_w + dfw] *
                                         descriptors_output[dfc * desc_fea_w * desc_fea_h + dfh * desc_fea_w + dfw];
            }
            float desc_channel_sum_sqrt = std::sqrt(desc_channel_sum_temp);
            desc_channel_sum_sqrts[dfh * desc_fea_w + dfw] = desc_channel_sum_sqrt;
            for (int dfc = 0; dfc < desc_fea_c; dfc++)
            {
                descriptors_output[dfc * desc_fea_w * desc_fea_h + dfh * desc_fea_w + dfw] =
                    descriptors_output[dfc * desc_fea_w * desc_fea_h + dfh * desc_fea_w + dfw] / desc_channel_sum_sqrts[dfh * desc_fea_w + dfw];
            }
        }
    }
    int s = 8;
    float *descriptors_output_f = new float[desc_fea_c * vdt_size];
    float *descriptors_output_sqrt = new float[vdt_size];
    int count = 0;

    // Interpolation
    // "256 * H/8 * W/8 to a full image descriptor of 256 * H * W"
    for (auto _vdt : keypoints)
    {
        float ix = ((_vdt.pt.x - s / 2 + 0.5) / (desc_fea_w * s - s / 2 - 0.5)) * (desc_fea_w - 1);
        float iy = (_vdt.pt.y - s / 2 + 0.5) / (desc_fea_h * s - s / 2 - 0.5) * (desc_fea_h - 1);

        int ix_nw = std::floor(ix);
        int iy_nw = std::floor(iy);

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;

        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;

        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        float descriptors_channel_sum_l2 = 0.f;
        for (int dfc = 0; dfc < desc_fea_c; dfc++)
        {
            float res = 0.f;

            if (lm_tools::within_bounds_2d(iy_nw, ix_nw, desc_fea_h, desc_fea_w))
            {
                res += descriptors_output[dfc * desc_fea_h * desc_fea_w + iy_nw * desc_fea_w + ix_nw] * nw;
            }
            if (lm_tools::within_bounds_2d(iy_ne, ix_ne, desc_fea_h, desc_fea_w))
            {
                res += descriptors_output[dfc * desc_fea_h * desc_fea_w + iy_ne * desc_fea_w + ix_ne] * ne;
            }
            if (lm_tools::within_bounds_2d(iy_sw, ix_sw, desc_fea_h, desc_fea_w))
            {
                res += descriptors_output[dfc * desc_fea_h * desc_fea_w + iy_sw * desc_fea_w + ix_sw] * sw;
            }
            if (lm_tools::within_bounds_2d(iy_se, ix_se, desc_fea_h, desc_fea_w))
            {
                res += descriptors_output[dfc * desc_fea_h * desc_fea_w + iy_se * desc_fea_w + ix_se] * se;
            }
            descriptors_output_f[dfc * vdt_size + count] = res;
            descriptors_channel_sum_l2 += res * res;
        }
        descriptors_output_sqrt[count] = descriptors_channel_sum_l2;
        for (int64_t dfc = 0; dfc < desc_fea_c; dfc++)
        {
            descriptors_output_f[dfc * vdt_size + count] /= std::sqrt(descriptors_output_sqrt[count]);
        }
        count++;
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < 4; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaFreeHost(scores_output);
    cudaFreeHost(descriptors_output);

    // delete[] scores_output;
    // delete[] descriptors_output;
    // scores_output = nullptr;
    // descriptors_output = nullptr;
    delete[] descriptors_output_sqrt;
    descriptors_output_sqrt = nullptr;

    // dp.descriptors = descriptors_output_f;
    // dp.desc_h = desc_fea_c;
    // dp.desc_w = vdt_size;
    // dp.status_code = 1;
    // return 1;

    // channels * number of keypoints
    descriptors = cv::Mat(desc_fea_c, vdt_size, CV_32F, descriptors_output_f);
    // Transpose descriptors to get number of keypoints * channels
    // ORBSLAM-3 expects it to be in this format
    cv::transpose(descriptors, descriptors);

    return 1;
}

SuperPoint::SuperPoint()
{
    cudaSetDevice(0);
    _point_lm_params.input_names.push_back("input");
    _point_lm_params.output_names.push_back("scores");
    _point_lm_params.output_names.push_back("descriptors");
    _point_lm_params.dla_core = -1;
    _point_lm_params.int8 = false;
    _point_lm_params.fp16 = false;
    _point_lm_params.batch_size = 1;
    _point_lm_params.seria = false;
    _point_lm_params.point_weight_file = "engines/superpoint_v1.engine";
    // _point_lm_params.input_h = 240;
    // _point_lm_params.input_w = 320;
    _point_lm_params.scores_thresh = 0.01;
    _point_lm_params.border = 4;
}

SuperPoint::~SuperPoint()
{
}

// int SuperPoint::get_input_h()
// {
//     return _point_lm_params.input_h;
// }

// int SuperPoint::get_input_w()
// {
//     return _point_lm_params.input_w;
// }

float *SuperPoint::imnormalize(cv::Mat &img)
{
    int img_h = img.rows;
    int img_w = img.cols;
    float *blob = new float[img_h * img_w];
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            blob[img_w * h + w] = ((float)img.at<uchar>(h, w)) / 255.f;
        }
    }
    return blob;
}

void SuperPoint::imnormalize(cv::Mat &img, float *blob)
{
    int img_h = img.rows;
    int img_w = img.cols;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            blob[img_w * h + w] = ((float)img.at<uchar>(h, w)) / 255.f;
        }
    }
}