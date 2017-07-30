
#include "../util/Math.hpp"
#include "PythonContext.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

PythonMainContext &GlobalContext(void);
void Initialise(void);

bp::object Import(const std::string &module, const std::string &path,
                  bp::object &globals);

std::string ParseException(void);

template <typename T>
inline std::vector<T> ToStdVector(const bp::object &iterable) {
  return std::vector<T>(bp::stl_input_iterator<T>(iterable),
                        bp::stl_input_iterator<T>());
}

np::ndarray ToNumpy(const EMatrix &mat);
np::ndarray ToNumpy(const EVector &vec);

template <typename T> inline np::ndarray ToNumpy(const std::vector<T> &vec) {
  bp::tuple shape = bp::make_tuple(vec.size());
  bp::tuple stride = bp::make_tuple(sizeof(T));
  return np::from_data(vec.data(), np::dtype::get_builtin<T>(), shape, stride,
                       bp::object())
      .copy();
}

template <> inline np::ndarray ToNumpy(const std::vector<bool> &vec) {
  bp::tuple shape = bp::make_tuple(vec.size());
  bp::tuple stride = bp::make_tuple(sizeof(bool));

  // Because vector<bool> packs the bools into bits.
  bool *normData = new bool[vec.size()];
  for (unsigned i = 0; i < vec.size(); i++) {
    normData[i] = vec[i];
  }

  np::ndarray result = np::from_data(normData, np::dtype::get_builtin<bool>(),
                                     shape, stride, bp::object())
                           .copy();
  delete[] normData;
  return result;
}

EVector ToEigen1D(const np::ndarray &arr);
EMatrix ToEigen2D(const np::ndarray &arr);
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array);
