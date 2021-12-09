#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
// #include "calibrator.h"

#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int IMG_H = Yolo::IMG_H;
static const int IMG_W = Yolo::IMG_W;
// static const int CLASS_NUM = 1;

static const int OUTPUT_DET_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
static const int OUTPUT_DA_SIZE = Yolo::IMG_H * Yolo::IMG_W;
static const int OUTPUT_LANE_SIZE = Yolo::IMG_H * Yolo::IMG_W;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_DET_NAME = "det_prob";
const char* OUTPUT_DA_NAME = "drivable_mask";
const char* OUTPUT_LANE_NAME = "lane_line_mask";

static Logger gLogger;

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    //create input tensor
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /*-----yoloP backbone------*/
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0"); //output 593
    auto convblock_50 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1"); //output 603
    auto bottleneck_csp1 = bottleneckCSP(network, weightMap, *convblock_50->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2"); //output 649
    auto convblock_101 = convBlock(network, weightMap, *bottleneck_csp1->getOutput(0),128, 3, 2, 1, "model.3"); //output 659
    auto bottleneck_csp2 = bottleneckCSP(network, weightMap, *convblock_101->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4"); //output 747 : 2output, concat with 1001, next
    auto convblock_190 = convBlock(network, weightMap, *bottleneck_csp2->getOutput(0), 256, 3, 2, 1, "model.5"); //output 757 : 2output, concat with 831,  next
    auto bottleneck_csp3 = bottleneckCSP(network, weightMap, *convblock_190->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6"); //output 845: 2output, concat with 939, next
    auto convblock279 = convBlock(network, weightMap, *bottleneck_csp3->getOutput(0), 512, 3, 2, 1, "model.7"); //output855
    auto spp = SPP(network, weightMap, *convblock279->getOutput(0), 512, 512, 5, 9, 13, "model.8");//output879


    auto convblock310 = convBlock(network, weightMap, *spp->getOutput(0), 256, 1, 1, 1, "model.9.cv1"); //output889
    auto convblock319 = convBlock(network, weightMap, *convblock310->getOutput(0), 256, 1, 1, 1, "model.9.m.0.cv1");//output899
    auto convblock328 = convBlock(network, weightMap, *convblock319->getOutput(0), 256, 3, 1, 1, "model.9.m.0.cv2"); //output 909

    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    auto conv_337 = network->addConvolutionNd(*convblock328->getOutput(0), 256, DimsHW{1, 1}, weightMap["model.9.cv3.weight"], emptywts);
    auto conv_338 = network->addConvolutionNd(*spp->getOutput(0), 256, DimsHW{1, 1}, weightMap["model.9.cv2.weight"], emptywts);
    ITensor* temp_cat339_tensor[] = { conv_337->getOutput(0), conv_338->getOutput(0) };
    IConcatenationLayer* cat_339 = network->addConcatenation(temp_cat339_tensor, 2);
    IScaleLayer* bn_340 = addBatchNorm2d(network, weightMap, *cat_339->getOutput(0), "model.9.bn", 1e-4);
    auto leakyrelu_341 = network->addActivation(*bn_340->getOutput(0), ActivationType::kLEAKY_RELU);
    leakyrelu_341->setAlpha(0.1);
    auto convblock_342 = convBlock(network, weightMap, *leakyrelu_341->getOutput(0), 512, 1, 1, 1, "model.9.cv4");
    auto convblock_351 = convBlock(network, weightMap, *convblock_342->getOutput(0), 256, 1, 1, 1, "model.10");
    
    //conv 351 resizE to cat with bottleneck_csp3 (outputid 845)output
    IResizeLayer *upsample_361 = network->addResize(*convblock_351->getOutput(0));
    Dims temp_upsample361_dim = bottleneck_csp3->getOutput(0)->getDimensions();
    upsample_361->setResizeMode(ResizeMode::kNEAREST);
    upsample_361->setOutputDimensions(temp_upsample361_dim); //need Dims3 {256, 40, 40}
    // upsample_361->setAlignCorners(true); // tips!
    ITensor* temp_cat362_tensor[] = { upsample_361->getOutput(0) , bottleneck_csp3->getOutput(0)};
    IConcatenationLayer* cat_362 = network->addConcatenation(temp_cat362_tensor, 2);

    // Dimension_checker(*cat_362->getOutput(0), "cat_362");

    /****************conv363, conv372, conv381  conv390conv391 -> cat****************/
    auto convblock363 = convBlock(network, weightMap, *cat_362->getOutput(0), 128, 1, 1, 1, "model.13.cv1"); //output889
    // Dims convblock363_dim = convblock363->getOutput(0)->getDimensions();
    // std::cout << convblock363_dim.d[0] << " " << convblock363_dim.d[1] << " " << convblock363_dim.d[2] << " " << convblock363_dim.d[3] << std::endl; 
    auto convblock372 = convBlock(network, weightMap, *convblock363->getOutput(0), 128, 1, 1, 1, "model.13.m.0.cv1");//output899
    auto convblock381 = convBlock(network, weightMap, *convblock372->getOutput(0), 128, 3, 1, 1, "model.13.m.0.cv2"); //output 909
    Weights emptywts_conv390391{ DataType::kFLOAT, nullptr, 0 };
    auto conv_390 = network->addConvolutionNd(*convblock381->getOutput(0), 128, DimsHW{1, 1}, weightMap["model.13.cv3.weight"], emptywts_conv390391);
    auto conv_391 = network->addConvolutionNd(*cat_362->getOutput(0), 128, DimsHW{1, 1}, weightMap["model.13.cv2.weight"], emptywts_conv390391);
    ITensor* temp_cat392_tensor[] = {conv_390->getOutput(0),  conv_391->getOutput(0)};
    IConcatenationLayer* cat_392 = network->addConcatenation(temp_cat392_tensor, 2);
    IScaleLayer* bn_393 = addBatchNorm2d(network, weightMap, *cat_392->getOutput(0), "model.13.bn", 1e-3);
    auto leakyrelu_394 = network->addActivation(*bn_393->getOutput(0), ActivationType::kLEAKY_RELU);
    leakyrelu_394->setAlpha(0.1);
    auto convblock_395 = convBlock(network, weightMap, *leakyrelu_394->getOutput(0), 256, 1, 1, 1, "model.13.cv4");


    auto convblock_404 = convBlock(network, weightMap, *convblock_395->getOutput(0), 128, 1, 1, 1, "model.14");
    // Dimension_checker(*convblock_404->getOutput(0), "convblock_404");
    //resize and concate


    IResizeLayer *upsample_414 = network->addResize(*convblock_404->getOutput(0));
    Dims temp_upsample414_dim = bottleneck_csp2->getOutput(0)->getDimensions();
    upsample_414->setResizeMode(ResizeMode::kNEAREST);
    upsample_414->setOutputDimensions(temp_upsample414_dim); //need Dims3 {256, 40, 40}
    // upsample_414->setAlignCorners(true); // tips!
    ITensor* temp_cat415_tensor[] = { upsample_414->getOutput(0), bottleneck_csp2->getOutput(0)};
    IConcatenationLayer* cat_415 = network->addConcatenation(temp_cat415_tensor, 2);
    Dimension_checker(*cat_415->getOutput(0), "cat_415");

    /**convblock 1562, convblock416, convblock1687**/
    auto convblock_1562 = convBlock(network, weightMap, *cat_415->getOutput(0), 128, 3, 1, 1, "model.25");
    Dimension_checker(*convblock_1562->getOutput(0), "convblock_1562");
    // auto resize_1572 = addResize(network, *convblock_1562->getOutput(0), 2.0);
    auto resize_1572 = addResize(network, *convblock_1562->getOutput(0), Dims3{128, INPUT_H/4, INPUT_W/4});

    auto convblock_1687 = convBlock(network, weightMap, *cat_415->getOutput(0), 128, 3, 1, 1, "model.34");
    Dimension_checker(*convblock_1687->getOutput(0), "convblock_1687");
    // auto resize_1697 = addResize(network, *convblock_1687->getOutput(0), 2.0);
    auto resize_1697 = addResize(network, *convblock_1687->getOutput(0), Dims3{128, INPUT_H/4, INPUT_W/4});


    //27 and 36 dict
    auto convblock_1605 = multi_convBlocks(network, weightMap, *resize_1572->getOutput(0), 128, 4, "model.27"); //output64 it will be drivable segmentation
    Dimension_checker(*convblock_1605->getOutput(0), "convblock_1605");

    auto convblock_1730 = multi_convBlocks(network, weightMap, *resize_1697->getOutput(0), 128, 4, "model.36");//output64 it will be lane line
    auto convblock_448 = multi_convBlocks(network, weightMap, *cat_415->getOutput(0), 256, 4, "model.17"); //key:model.17 //output128
    Dimension_checker(*convblock_448->getOutput(0), "convblock_448");

    //28 and 37 dict
    auto convblock_1614 = convBlock(network, weightMap, *convblock_1605->getOutput(0), 32, 3, 1, 1, "model.28");
    Dimension_checker(*convblock_1614->getOutput(0), "convblock_1614");
    // auto resize_1624 = addResize(network, *convblock_1614->getOutput(0), 2.0);
    auto resize_1624 = addResize(network, *convblock_1614->getOutput(0), Dims3{32, INPUT_H/2, INPUT_W/2});

    auto convblock_1625 = convBlock(network, weightMap, *resize_1624->getOutput(0), 16, 3, 1, 1, "model.30");
    auto convblock_1666 = multi_convBlocks(network, weightMap, *convblock_1625->getOutput(0), 16, 4, "model.31"); //output64


    auto convblock_1739 = convBlock(network, weightMap, *convblock_1730->getOutput(0), 32, 3, 1, 1, "model.37");
    Dimension_checker(*convblock_1739->getOutput(0), "convblock_1739");
    // auto resize_1749 = addResize(network, *convblock_1739->getOutput(0), 2.0);
    auto resize_1749 = addResize(network, *convblock_1739->getOutput(0), Dims3{32, INPUT_H/2, INPUT_W/2});

    auto convblock_1750 = convBlock(network, weightMap, *resize_1749->getOutput(0), 16, 3, 1, 1, "model.39");

    auto convblock_1791 = multi_convBlocks(network, weightMap, *convblock_1750->getOutput(0), 16, 4, "model.40"); //output64

    //--------------33 and 42 segmentatiob and lane line 
    Dimension_checker(*convblock_1666->getOutput(0), "convblock_1666");
    // auto resize_1676 = addResize(network, *convblock_1666->getOutput(0), 2.0);
    auto resize_1676 = addResize(network, *convblock_1666->getOutput(0), Dims3{8, INPUT_H, INPUT_W});

    auto convblock_1677 = convBlock(network, weightMap, *resize_1676->getOutput(0), 2, 3, 1, 1, "model.33"); //da
    Dimension_checker(*convblock_1791->getOutput(0), "convblock_1791");
    // auto resize_1801 = addResize(network, *convblock_1791->getOutput(0), 2.0);
    auto resize_1801 = addResize(network, *convblock_1791->getOutput(0), Dims3{8, INPUT_H, INPUT_W});
    auto convblock_1802 = convBlock(network, weightMap, *resize_1801->getOutput(0), 2, 3, 1, 1, "model.42"); //lane
    // auto sig_1686 = network->addActivation(*convblock_1677->getOutput(0), ActivationType::kSIGMOID);
    // auto sig_1811 = network->addActivation(*convblock_1802->getOutput(0), ActivationType::kSIGMOID);
    ISliceLayer *laneSlice = network->addSlice(*convblock_1802->getOutput(0), Dims3{ 0, (Yolo::INPUT_H - IMG_H) / 2, 0 }, Dims3{ 2, IMG_H, IMG_W  }, Dims3{ 1, 1, 1 }); //lane
    auto lane_out = network->addTopK(*laneSlice->getOutput(0), TopKOperation::kMAX, 1, 1);
    Dimension_checker(*lane_out->getOutput(0), "lane_out");
    ISliceLayer *daSlice = network->addSlice(*convblock_1677->getOutput(0), Dims3{ 0, (Yolo::INPUT_H - IMG_H) / 2, 0 }, Dims3{ 2, IMG_H, IMG_W }, Dims3{ 1, 1, 1 }); //da
    auto da_out = network->addTopK(*daSlice->getOutput(0), TopKOperation::kMAX, 1, 1);
    Dimension_checker(*da_out->getOutput(0), "da_out");

    //----------- detection---------------//
    auto convblock_457 = convBlock(network, weightMap, *convblock_448->getOutput(0), 128, 3, 2, 1, "model.18");
    Dimension_checker(*convblock_457->getOutput(0), "convblock_457");
    ITensor* temp_cat466_tensor[] = {convblock_457->getOutput(0),  convblock_404->getOutput(0) }; //404 cat with 457
    IConcatenationLayer* cat_466 = network->addConcatenation(temp_cat466_tensor, 2);
    Dimension_checker(*cat_466->getOutput(0), "cat_466");
    auto convblock_499 = multi_convBlocks(network, weightMap, *cat_466->getOutput(0), 256, 2, "model.20"); //output64
    auto convblock_508 = convBlock(network, weightMap, *convblock_499->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* temp_cat517_tensor[] = {convblock_508->getOutput(0),  convblock_351->getOutput(0)};
    IConcatenationLayer* cat_517 = network->addConcatenation(temp_cat517_tensor, 2);
    auto convblock_550 = multi_convBlocks(network, weightMap, *cat_517->getOutput(0), 512, 2, "model.23"); //output64
    Weights emptys{ DataType::kFLOAT, nullptr, 0 };
    // auto conv_559 = network->addConvolutionNd(*convblock_448->getOutput(0), 18, DimsHW{1, 1}, weightMap["model.24.m.0.weight"], emptys);
    auto det0 = network->addConvolutionNd(*convblock_448->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    // Dimension_checker(*det0->getOutput(0), "det0");
    // auto conv_893 = network->addConvolutionNd(*convblock_499->getOutput(0), 18, DimsHW{1, 1}, weightMap["model.24.m.1.weight"], emptys);
    auto det1 = network->addConvolutionNd(*convblock_499->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    // Dimension_checker(*det1->getOutput(0), "det1");

    // auto sig_946 = reshape_transpose_sigmoid(network, *conv_893->getOutput(0), 1600, 40);

    // auto conv_1227 = network->addConvolutionNd(*convblock_550->getOutput(0), 18, DimsHW{1, 1}, weightMap["model.24.m.2.weight"], emptys);
    auto det2 = network->addConvolutionNd(*convblock_550->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
    // Dimension_checker(*det2->getOutput(0), "det2");


    // auto yolo_det = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    auto yolo_det = addYoLoLayer(network, weightMap, det0, det1, det2);
    // Dimension_checker(*yolo_det->getOutput(0), "yolo_det");

    //detection result
    yolo_det->getOutput(0) -> setName(OUTPUT_DET_NAME);
    network->markOutput(*yolo_det->getOutput(0));
    // convblock363->getOutput(0) -> setName(OUTPUT_DET_NAME);
    // network->markOutput(*convblock363->getOutput(0));

    //drivable mask
    da_out->getOutput(1) -> setName(OUTPUT_DA_NAME);
    network->markOutput(*da_out->getOutput(1));

    //lane line mask
    lane_out->getOutput(1) -> setName(OUTPUT_LANE_NAME);
    network->markOutput(*lane_out->getOutput(1));
    

    //Builder engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(2L * (1L << 30));
    // config->setMaxWorkspaceSize( (1 << 30));

    std::cout << "Building YOLOP engine, please wait for a while.." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build yoloP sucessfully !" << std::endl;
    network->destroy();
    for (auto&mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return engine;

}


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


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    
    engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

int main(int argc, char** argv){
    cudaSetDevice(DEVICE);
    std::string wts_name = "output.wts";
    std::string engine_name = "/home/liwei.fang/YOLOP-main/tensorrt_version/test_2080.engine";

    // IHostMemory* modelStream{ nullptr };
    // APIToModel(BATCH_SIZE, &modelStream, wts_name);
    // //loading model
    // assert(modelStream != nullptr);
    // std::cout << "haha" << std::endl;
    // std::ofstream p(engine_name, std::ios::binary);
    // if (!p) {
    //     std::cerr << "could not open plan output file" << std::endl;
    //     return -1;
    // }

    // p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    // modelStream->destroy();

    //deserialize the .engine and run inference
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

    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    //b
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_DET_SIZE]; //detection output
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
    // const char img_dir[2] = "/home/stephen/VScodeProjects/yolop-tensorrt/examples";
    std::string img_dir = "/home/liwei.fang/YOLOP-main/tensorrt_version/examples";
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    };


   // store seg results
    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, drivable_mask);
    // sotore lane results
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_mask);
    cv::Mat seg_res(720, 1280, CV_32S);
    cv::Mat lane_res(720, 1280, CV_32S);


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
                std::cout << "draw";
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }
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
}
