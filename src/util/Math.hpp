#pragma once

#include <Eigen/Dense>
#include <vector>

using EVector = Eigen::VectorXf;
using EMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace util {

double RandInterval(double s, double e);
double GaussianSample(double mean, double sd);

std::vector<float> SoftmaxWeights(const std::vector<float> &in);
float SoftmaxWeightedAverage(const std::vector<float> &in, float temperature);
}
