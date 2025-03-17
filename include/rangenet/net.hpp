/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once

// standard stuff
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// opencv
#include <opencv2/core/core.hpp>

// yamlcpp
#include <yaml-cpp/yaml.h>

namespace rangenet {

/**
 * @brief      Class for segmentation network inference.
 */
class Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   */
  explicit Net(const std::string& model_path);

  /**
   * @brief      Destroys the object.
   */
  virtual ~Net() = default;

  /**
   * @brief      Infer logits from LiDAR scan
   *
   * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
   *
   * @return     Semantic estimates with probabilities over all classes (_n_classes, _img_h, _img_w)
   */
  virtual std::vector<std::vector<float>> infer(
      const std::vector<float>& scan, const uint32_t& num_points) = 0;

  /**
   * @brief      Get raw point clouds
   *
   * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
   *
   * @return     cv format points
   */
  std::vector<cv::Vec3f> getPoints(
      const std::vector<float>& scan, const uint32_t& num_points);

  /**
   * @brief      Convert mask to color using dictionary as lut
   *
   * @param[in]  semantic_scan, The mask from argmax; num_points, the number of points in this scan.
   *
   * @return     the colored segmentation mask :)
   */
  void getLabels(const std::vector<std::vector<float>>& semantic_scan,
                 std::vector<std::string>& labels,
                 std::vector<cv::Vec3b>& colors);
  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose) { verbose_ = verbose; }

  /**
   * @brief      Get the label map
   *
   * @return     the learning label mapping
   */
  std::map<uint32_t, std::string> getLabelMap() const { return label_map_; }

  /**
   * @brief      Get the color map
   *
   * @return     the color map
   */
  std::map<uint32_t, cv::Vec3b> getColorMap() const { return color_map_; }

 protected:
  // general
  std::string model_path_;  // Where to get model weights and cfg
  bool verbose_;            // verbose mode?

  // image properties
  int img_h_, img_w_, img_d_;     // height, width, and depth for inference
  std::vector<float> img_means_;  // mean and std per channel
  std::vector<float> img_stds_;   // mean and std per channel
  // problem properties
  int32_t n_classes_;  // number of classes to differ from
  // sensor properties
  double fov_up_;    // field of view up and down in radians
  double fov_down_;  // field of view up and down in radians

  // config
  YAML::Node data_cfg_;  // yaml nodes with configuration from training
  YAML::Node arch_cfg_;  // yaml nodes with configuration from training

  std::map<uint32_t, std::string> label_map_;
  std::map<uint32_t, std::string> learning_label_map_;  // for color conversion
  std::map<uint32_t, cv::Vec3b> color_map_;
  std::map<uint32_t, cv::Vec3b> learning_color_map_;  // for color conversion
};

/**
 * @brief Makes a network with the desired backend, checking that it exists,
 *        it is implemented, and that it was compiled.
 *
 * @param[in] path The path to the model directory
 *
 * @return std::unique_ptr<Net>
 */
std::unique_ptr<Net> make_net(const std::string& path);

}  // namespace rangenet
