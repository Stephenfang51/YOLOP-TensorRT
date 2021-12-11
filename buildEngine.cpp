#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.hpp"
#include "cxxopts.hpp"
#include "network.hpp"

#define DEVICE 0

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
    cxxopts::Options options("YOLOP-TensorRT", "using TensorRT to speed up your YOLOP model");
    options.add_options()
            ("w,wts", "enter your wts path", cxxopts::value<std::string>()->default_value("yolop.wts"))
            ("b,batchsize", "enter batchsize, default", cxxopts::value<int>()->default_value("1"))
            ("o,output", "enter your output path", cxxopts::value<std::string>()->default_value("yolop.engine"))
    ;

    auto result = options.parse(argc, argv);
    cudaSetDevice(DEVICE);
    if (result.count("w")){
        std::string wts_name = result["w"].as<std::string>();
    }
    else{
        std::cerr << "please specify your wts" << std::endl;
        std::exit;
    };

    std::string wts_name = result["w"].as<std::string>();
    std::string engine_name = result["output"].as<std::string>();
    int batchsize = result["b"].as<int>();

    IHostMemory* modelStream{ nullptr };
    APIToModel(batchsize, &modelStream, wts_name);
    //loading model
    assert(modelStream != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p){
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }

    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    }

