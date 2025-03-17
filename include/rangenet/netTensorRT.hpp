/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <chrono>
#include <numeric>
#include "net.hpp"

// GPU workspace size configs
#define MAX_WORKSPACE_SIZE (1UL << 33)  // 8GB
#define MIN_WORKSPACE_SIZE (1UL << 20)  // 1MB

// DLA configs
#define DEVICE_DLA_0 0  // DLA core 0
#define DEVICE_DLA_1 0  // DLA core 1

using namespace nvinfer1;

#define CUDA_CHECK(status)                                                      \
  if (status != cudaSuccess) {                                                  \
    printf("%s in %s at %d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
    exit(-1);                                                                   \
  }

namespace rangenet {

// TensorRT logger
class Logger : public ILogger {
 public:
  void set_verbosity(bool verbose) { verbose_ = verbose; }
  void log(Severity severity, const char* msg) noexcept {
    if (verbose_) {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "INTERNAL_ERROR: ";
          break;
        case Severity::kERROR:
          std::cerr << "ERROR: ";
          break;
        case Severity::kWARNING:
          std::cerr << "WARNING: ";
          break;
        case Severity::kINFO:
          std::cerr << "INFO: ";
          break;
        default:
          std::cerr << "UNKNOWN: ";
          break;
      }
      std::cout << msg << std::endl;
    }
  }

 private:
  bool verbose_ = false;
};

/**
 * @brief TensorRT inference network
 */
class NetTensorRT : public Net {
 public:
  explicit NetTensorRT(const std::string& model_path);
  ~NetTensorRT();

  std::vector<std::vector<float>> infer(
      const std::vector<float>& scan,
      const uint32_t& num_points) override;

  void verbosity(const bool verbose);

 protected:
  // TensorRT components
  ICudaEngine* engine_;
  IExecutionContext* context_;
  Logger gLogger_;
  std::vector<void*> device_buffers_;
  std::vector<void*> host_buffers_;
  cudaStream_t cuda_stream_;
  uint in_bind_idx_;
  uint out_bind_idx_;

  // Projection data
  std::vector<float> proj_xs_;
  std::vector<float> proj_ys_;

  // Invalid point markers
  const std::vector<float> invalid_input_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> invalid_output_ = std::vector<float>(20, 0.0f);

  // Timer
  std::vector<std::chrono::system_clock::time_point> stimes_;

  // Helper functions
  template <typename T>
  std::vector<size_t> sort_indexes(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
  }

  std::vector<std::vector<float>> doProjection(
      const std::vector<float>& scan,
      const uint32_t& num_points);

  int getBufferSize(Dims d, DataType t);
  void deserializeEngine(const std::string& engine_path);
  void serializeEngine(const std::string& engine_path);
  void generateEngine(const std::string& onnx_path);
  void prepareBuffer();

  // Timer utilities
  void tic() {
    stimes_.push_back(std::chrono::high_resolution_clock::now());
  }
  double toc() {
    assert(stimes_.begin() != stimes_.end());

    std::chrono::system_clock::time_point endtime = std::chrono::high_resolution_clock::now();
    std::chrono::system_clock::time_point starttime = stimes_.back();
    stimes_.pop_back();

    std::chrono::duration<double> elapsed_seconds = endtime - starttime;

    return elapsed_seconds.count();
  }
};

}  // namespace rangenet
