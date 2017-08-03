
#include "Math.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

double util::RandInterval(double s, double e) {
  return s + (e - s) * (rand() / (double)RAND_MAX);
}

double util::GaussianSample(double mean, double sd) {
  // Taken from GSL Library Gaussian random distribution.
  double x, y, r2;

  do {
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = -1 + 2 * RandInterval(0.0, 1.0);
    y = -1 + 2 * RandInterval(0.0, 1.0);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0);

  // Box-Muller transform
  return mean + sd * y * sqrt(-2.0 * log(r2) / r2);
}

std::vector<float> util::SoftmaxWeights(const std::vector<float> &in) {
  assert(in.size() > 0);

  std::vector<float> result(in.size());

  float maxVal = in[0];
  for (unsigned r = 0; r < in.size(); r++) {
    maxVal = fmax(maxVal, in[r]);
  }

  float sum = 0.0f;
  for (unsigned i = 0; i < in.size(); i++) {
    result[i] = expf(in[i] - maxVal);
    sum += result[i];
  }

  for (unsigned i = 0; i < result.size(); i++) {
    result[i] /= sum;
  }

  return result;
}

float util::SoftmaxWeightedAverage(const std::vector<float> &in,
                                   float temperature) {
  assert(temperature > 0.0f);

  std::vector<float> tempAdjusted(in.size());
  for (unsigned i = 0; i < in.size(); i++) {
    tempAdjusted[i] = in[i] / temperature;
  }

  auto weights = SoftmaxWeights(tempAdjusted);

  float result = 0.0f;
  for (unsigned i = 0; i < in.size(); i++) {
    result += weights[i] * in[i];
  }
  return result;
}

unsigned util::SoftmaxSample(const std::vector<float> &rawWeights,
                             float temperature) {
  assert(rawWeights.size() > 0);
  assert(temperature > 0.0f);

  std::vector<float> adjusted;
  for (auto rw : rawWeights) {
    adjusted.emplace_back(rw / temperature);
  }

  adjusted = SoftmaxWeights(adjusted);
  float s = util::RandInterval(0.0f, 1.0f);

  for (unsigned i = 0; i < adjusted.size(); i++) {
    if (adjusted[i] >= s) {
      return i;
    }
    s -= adjusted[i];
  }

  return rawWeights.size() - 1;
}

unsigned util::SampleFromDistribution(const std::vector<float> &weights) {
  assert(weights.size() > 0);

  float sumWeights = 0.0f;
  for (auto w : weights) {
    sumWeights += w;
  }

  float s = util::RandInterval(0.0f, sumWeights);

  // for (auto w : weights) {
  //   std::cout << " " << w;
  // }
  // std::cout << std::endl;

  for (unsigned i = 0; i < weights.size(); i++) {
    if (weights[i] >= s) {
      return i;
    }
    s -= weights[i];
  }

  return weights.size() - 1;
}
