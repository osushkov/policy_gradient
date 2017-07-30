
#include "PythonUtil.hpp"
#include "../util/Common.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <mutex>

using namespace std;

static std::once_flag initialiseFlag;
static uptr<python::PythonMainContext> globalContext;

python::PythonMainContext &python::GlobalContext(void) {
  return *globalContext;
}

void python::Initialise(void) {
  std::call_once(initialiseFlag, []() {
    Py_Initialize();
    PyEval_InitThreads();
    np::initialize();

    globalContext = make_unique<PythonMainContext>();
  });
}

bp::object python::Import(const std::string &module, const std::string &path,
                          bp::object &globals) {
  bp::dict locals;
  locals["module_name"] = module;
  locals["path"] = path;

  bp::exec("import imp\n"
           "import sys\n"
           "sys.path.append('src/python')\n"
           "new_module = imp.load_module(module_name, open(path), path, ('py', "
           "'U', imp.PY_SOURCE))\n",
           globals, locals);
  return locals["new_module"];
}

std::string python::ParseException(void) {
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  std::string ret("Unfetchable Python error");

  if (type != nullptr) {
    bp::handle<> hType(type);
    bp::str typeStr(hType);
    bp::extract<std::string> eTypeStr(typeStr);
    ret = eTypeStr.check() ? eTypeStr() : "Unknown exception type";
  }

  if (value != nullptr) {
    bp::handle<> hVal(value);
    bp::str a(hVal);
    bp::extract<std::string> returned(a);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python error: ");
  }

  if (traceback != nullptr) {
    bp::handle<> hTb(traceback);
    bp::object tb(bp::import("traceback"));
    bp::object fmtTb(tb.attr("format_tb"));
    bp::object tbList(fmtTb(hTb));
    bp::object tbStr(bp::str("\n").join(tbList));
    bp::extract<std::string> returned(tbStr);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python traceback");
  }

  return ret;
}

np::ndarray python::ToNumpy(const EMatrix &mat) {
  bp::tuple shape = bp::make_tuple(mat.rows(), mat.cols());
  bp::tuple stride = bp::make_tuple(mat.cols() * sizeof(float), sizeof(float));
  return np::from_data(mat.data(), np::dtype::get_builtin<float>(), shape,
                       stride, bp::object())
      .copy();
}

np::ndarray python::ToNumpy(const EVector &vec) {
  bp::tuple shape = bp::make_tuple(vec.rows());
  bp::tuple stride = bp::make_tuple(sizeof(float));
  return np::from_data(vec.data(), np::dtype::get_builtin<float>(), shape,
                       stride, bp::object())
      .copy();
}

EVector python::ToEigen1D(const np::ndarray &arr) {
  assert(arr.get_nd() == 1);

  int rows = arr.shape(0);
  int rowStride = arr.strides(0);

  EVector result(rows);
  for (int r = 0; r < rows; r++) {
    result(r) = *(float *)(arr.get_data() + r * rowStride);
  }
  return result;
}

EMatrix python::ToEigen2D(const np::ndarray &arr) {
  assert(arr.get_nd() == 2);

  int rows = arr.shape(0);
  int rowStride = arr.strides(0);

  int cols = arr.shape(1);
  int colStride = arr.strides(1);

  EMatrix result(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      result(r, c) = *(float *)(arr.get_data() + r * rowStride + c * colStride);
    }
  }

  return result;
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array) {
  stream << bp::extract<char const *>(bp::str(array));
  return stream;
}
