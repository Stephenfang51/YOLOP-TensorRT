#include <iostream>
#include "cxxopts.hpp"
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "yololayer.h"
#include "utils.hpp"
#include "network.hpp"


void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* det_output, int*damask_output, int*lanemask_output,  int batchSize, int det_index, int index_da, int index_lane) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpyAsync(det_output, buffers[det_index], batchSize * OUTPUT_DET_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(damask_output, buffers[index_da], batchSize * OUTPUT_DA_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(lanemask_output, buffers[index_lane], batchSize * OUTPUT_LANE_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

   
int main(int argc, char**argv){
    cxxopts::Options options("YOLOP-TensorRT", "inference with image or video");
    options.add_options()
        ("e,engine", "specify your engine path", cxxopts::value<std::string>())
        ("img", "enter image directory", cxxopts::value<std::string>())
        ("v,video", "enter a video", cxxopts::value<std::string>())
        ("s,show", "if show video result", cxxopts::value<bool>()->default_value("false"))
        ;
            
    //deserialize the .engine and run inference
    auto opt = options.parse(argc, argv);
    std::string engine_name = opt["engine"].as<std::string>();
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
 
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    //allowcates each output
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_DET_SIZE];
    static int drivable_mask[BATCH_SIZE * OUTPUT_DA_SIZE];
    static int lane_mask[BATCH_SIZE * OUTPUT_LANE_SIZE];

    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);
    void* buffers[4];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex_det = engine->getBindingIndex(OUTPUT_DET_NAME);
    const int outputIndex_da = engine->getBindingIndex(OUTPUT_DA_NAME);
    const int outputIndex_lane = engine->getBindingIndex(OUTPUT_LANE_NAME);

    //Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_det], BATCH_SIZE * OUTPUT_DET_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_da], BATCH_SIZE * OUTPUT_DA_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_lane], BATCH_SIZE * OUTPUT_LANE_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<std::string> file_names;
    std::string img_dir = opt["img"].as<std::string>();

    //check dir or video
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    };

    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, drivable_mask);
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_mask);
    cv::Mat seg_res(720, 1280, CV_32S);
    cv::Mat lane_res(720, 1280, CV_32S);


    if (opt.count("img")){
        int fcount = 0;
        for (int f = 0; f < (int)file_names.size(); f++) {
            fcount++;
            if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
            for (int b = 0; b < fcount; b++) {
                cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                if (img.empty()) continue;
                cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }

            // Run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, stream, buffers, data, prob, drivable_mask, lane_mask, BATCH_SIZE, outputIndex_det, outputIndex_da, outputIndex_lane);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                nms(res, &prob[b * OUTPUT_DET_SIZE], CONF_THRESH, NMS_THRESH);
            }
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                //std::cout << res.size() << std::endl;
                cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                cv::resize(tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
                cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);
                draw_mask(img, seg_res, lane_res, res);
                for (size_t j = 0; j < res.size(); j++) {

                    cv::Rect r = get_rect(img, res[j].bbox);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                };
                cv::imwrite("output_" + file_names[f - fcount + 1 + b], img);
            };
            fcount = 0;
        };

    }; //end of img processing
    if (opt.count("v")){
        cv::VideoCapture cap;
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::VideoWriter writer("output.mp4", codec, 25.0, cv::Size(1280, 720));
        cap.open(opt["video"].as<std::string>());
        if (!cap.isOpened()){
        std::cerr << "fail to decodec file" <<std::endl;
        return -1;
        }
        cv::namedWindow("YOLOP",cv::WINDOW_AUTOSIZE);
        // tep_seg;
        while(true){
            cv::Mat frame;
            cap>>frame;
            
            if (frame.empty()){
                std::cerr << "empty frame"<<std::endl;
            }
            else{
                cv::Mat pr_img = preprocess_img(frame, INPUT_W, INPUT_H); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
                // Run inference
                auto start = std::chrono::system_clock::now();
                doInference(*context, stream, buffers, data, prob, drivable_mask, lane_mask, BATCH_SIZE, outputIndex_det, outputIndex_da, outputIndex_lane);
                auto end = std::chrono::system_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
                std::vector<Yolo::Detection> res;
                nms(res, prob, CONF_THRESH, NMS_THRESH);

                // for (int b = 0; b < fcount; b++) {
                    // auto& res = batch_res[b];
                    //std::cout << res.size() << std::endl;
                    // cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                cv::resize(tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
                cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);
                draw_mask(frame, seg_res, lane_res, res);
                //draw every bbox
                for (size_t j = 0; j < res.size(); j++) {
                    cv::Rect r = get_rect(frame, res[j].bbox);
                    cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(frame, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                    }
                // frame.convertTo(output, CV_8UC3);
                writer.write(frame);
                };
                // cv::imshow("YOLOP", frame);
            };//end of while loop
        };//end of video processing

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex_det]));
    CHECK(cudaFree(buffers[outputIndex_da]));
    CHECK(cudaFree(buffers[outputIndex_lane]));

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    // return 0;


}//end of main

    