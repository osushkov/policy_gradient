
#pragma once

#include <iostream>

namespace python {

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  unsigned maxBatchSize;

  NetworkSpec() : numInputs(0), numOutputs(0), maxBatchSize(0) {}
  NetworkSpec(unsigned numInputs, unsigned numOutputs, unsigned maxBatchSize)
      : numInputs(numInputs), numOutputs(numOutputs),
        maxBatchSize(maxBatchSize) {}

  inline void Write(std::ostream &out) {
    out << numInputs << std::endl;
    out << numOutputs << std::endl;
    out << maxBatchSize << std::endl;
  }

  static NetworkSpec Read(std::istream &in) {
    NetworkSpec spec;
    in >> spec.numInputs;
    in >> spec.numOutputs;
    in >> spec.maxBatchSize;
    return spec;
  }
};
}
