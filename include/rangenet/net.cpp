/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#include "net.hpp"
#include "netTensorRT.hpp"
#include <opencv2/opencv.hpp>

namespace rangenet {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 */
Net::Net(const std::string& model_path)
    : model_path_(model_path), verbose_(false) {
  // Set verbosity
  verbosity(verbose_);

  // Load architecture config
  std::string arch_cfg_path = model_path_ + "/arch_cfg.yaml";
  try {
    arch_cfg_ = YAML::LoadFile(arch_cfg_path);
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Failed to load arch_cfg.yaml from " + arch_cfg_path);
  }

  // Load FOV parameters
  fov_up_ = arch_cfg_["dataset"]["sensor"]["fov_up"].as<double>();
  fov_down_ = arch_cfg_["dataset"]["sensor"]["fov_down"].as<double>();

  // Load data config
  std::string data_cfg_path = model_path_ + "/data_cfg.yaml";
  try {
    data_cfg_ = YAML::LoadFile(data_cfg_path);
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Can't open cfg.yaml from " + data_cfg_path);
  }

  // Get color mapping
  YAML::Node label_map;
  try {
    label_map = data_cfg_["labels"];
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Failed to load label_map from " + data_cfg_path);
  }

  // Generate label mapping fomr indices
  YAML::const_iterator it;
  for (it = label_map.begin(); it != label_map.end(); ++it) {
    int key = it->first.as<int>();
    label_map_[key] = label_map[key].as<std::string>();
  }

  // Get color mapping
  YAML::Node color_map;
  try {
    color_map = data_cfg_["color_map"];
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Failed to load color_map from " + data_cfg_path);
  }

  // Generate color mapping from indices
  for (it = color_map.begin(); it != color_map.end(); ++it) {
    int key = it->first.as<int>();
    cv::Vec3b color(static_cast<u_char>(color_map[key][0].as<unsigned int>()),
                    static_cast<u_char>(color_map[key][1].as<unsigned int>()),
                    static_cast<u_char>(color_map[key][2].as<unsigned int>()));
    color_map_[key] = color;
  }

  // Get learning class labels
  YAML::Node learning_class;
  try {
    learning_class = data_cfg_["learning_map_inv"];
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Failed to load learning_map_inv from " + data_cfg_path);
  }

  // Get number of classes
  n_classes_ = learning_class.size();

  // Remap the colormap lookup table
  for (it = learning_class.begin(); it != learning_class.end(); ++it) {
    int key = it->first.as<int>();
    learning_color_map_[key] = color_map_[learning_class[key].as<unsigned int>()];
    learning_label_map_[key] = label_map_[learning_class[key].as<unsigned int>()];
  }

  // Get image properties
  img_h_ = arch_cfg_["dataset"]["sensor"]["img_prop"]["height"].as<int>();
  img_w_ = arch_cfg_["dataset"]["sensor"]["img_prop"]["width"].as<int>();
  img_d_ = 5;  // range, x, y, z, remission

  // Get normalization parameters
  YAML::Node img_means, img_stds;
  try {
    img_means = arch_cfg_["dataset"]["sensor"]["img_means"];
    img_stds = arch_cfg_["dataset"]["sensor"]["img_stds"];
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Failed to load img_means/stds from config");
  }

  // Fill means and stds
  for (it = img_means.begin(); it != img_means.end(); ++it) {
    img_means_.push_back(it->as<float>());
  }
  for (it = img_stds.begin(); it != img_stds.end(); ++it) {
    img_stds_.push_back(it->as<float>());
  }
}

/**
 * @brief      Get raw point clouds
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     cv format points
 */
std::vector<cv::Vec3f> Net::getPoints(const std::vector<float>& scan, const uint32_t& num_points) {
  std::vector<cv::Vec3f> points;
  points.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    points[i] = cv::Vec3f(scan[4 * i], scan[4 * i + 1], scan[4 * i + 2]);
  }
  return points;
}

/**
 * @brief      Convert mask to color using dictionary as lut
 *
 * @param[in]  semantic_scan The mask from argmax; num_points, the number of points in this scan.
 *
 * @return     the colored segmentation mask :)
 */
void Net::getLabels(const std::vector<std::vector<float>>& semantic_scan,
                    std::vector<std::string>& labels,
                    std::vector<cv::Vec3b>& colors) {
  int num_points = semantic_scan.size();
  labels.resize(num_points);
  colors.resize(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    int label_index = 0;
    float prob = 0.f;
    for (int32_t j = 0; j < n_classes_; ++j) {
      if (prob <= semantic_scan[i][j]) {
        prob = semantic_scan[i][j];
        label_index = j;
      }
    }
    colors[i] = learning_color_map_[label_index];
    labels[i] = learning_label_map_[label_index];
  }
  return;
}

/**
 * @brief Makes a network with the desired backend, checking that it exists,
 *        it is implemented, and that it was compiled.
 *
 * @param[in]  path  The path to the model
 *
 * @return std::unique_ptr<Net>
 */
std::unique_ptr<Net> make_net(const std::string& path) {
  return std::unique_ptr<Net>(new NetTensorRT(path));
}

}  // namespace rangenet
