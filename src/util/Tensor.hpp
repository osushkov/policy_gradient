#pragma once

#include "Common.hpp"
#include "Math.hpp"

#include <iostream>

namespace util {

class Tensor {
public:
  static Tensor Read(std::istream &in);
  void Write(std::ostream &out);

  unsigned NumLayers(void) const;
  void AddLayer(const EMatrix &m);

  EMatrix &operator()(unsigned index);
  const EMatrix &operator()(unsigned index) const;

  Tensor operator*(const Tensor &t) const;
  Tensor operator+(const Tensor &t) const;
  Tensor operator-(const Tensor &t) const;
  Tensor operator*(float s) const;
  Tensor operator/(float s) const;

  Tensor &operator*=(const Tensor &t);
  Tensor &operator+=(const Tensor &t);
  Tensor &operator-=(const Tensor &t);
  Tensor &operator*=(float s);
  Tensor &operator/=(float s);

private:
  vector<EMatrix> data;
};
}
