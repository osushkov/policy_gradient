#include "TFLearner.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

namespace np = boost::python::numpy;
namespace bp = boost::python;

using namespace python;

class LearnerInstance {
public:
  LearnerInstance() = default;
  virtual ~LearnerInstance() = default;

  virtual void Learn(const QLearnBatch &batch) = 0;
  virtual void UpdateTargetParams(void) = 0;
  virtual np::ndarray QFunction(const np::ndarray &state) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void Learn(const QLearnBatch &batch) override {
    get_override("Learn")(batch);
  }

  void UpdateTargetParams(void) override {
    get_override("UpdateTargetParams")();
  }

  np::ndarray QFunction(const np::ndarray &state) override {
    return get_override("QFunction")(state);
  }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();

  bp::class_<NetworkSpec>("NetworkSpec")
      .def_readonly("numInputs", &NetworkSpec::numInputs)
      .def_readonly("numOutputs", &NetworkSpec::numOutputs)
      .def_readonly("maxBatchSize", &NetworkSpec::maxBatchSize);

  bp::class_<QLearnBatch>("QLearnBatch")
      .def_readonly("initialStates", &QLearnBatch::initialStates)
      .def_readonly("successorStates", &QLearnBatch::successorStates)
      .def_readonly("actionsTaken", &QLearnBatch::actionsTaken)
      .def_readonly("isEndStateTerminal", &QLearnBatch::isEndStateTerminal)
      .def_readonly("rewardsGained", &QLearnBatch::rewardsGained)
      .def_readonly("futureRewardDiscount", &QLearnBatch::futureRewardDiscount)
      .def_readonly("learnRate", &QLearnBatch::learnRate);

  bp::class_<PyLearnerInstance, boost::noncopyable>("LearnerInstance");
}

struct TFLearner::TFLearnerImpl {
  bp::object learner;

  TFLearnerImpl(const NetworkSpec &spec) {
    try {
      PyImport_AppendInittab("LearnerFramework", &initLearnerFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object learnerModule =
          python::Import("learner", "python_src/learner.py", globals);

      bp::object Learner = learnerModule.attr("Learner");
      learner = Learner(spec);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }

  void Learn(const QLearnBatch &batch) {
    try {
      learner.attr("Learn")(batch);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }

  void UpdateTargetParams(void) {
    try {
      learner.attr("UpdateTargetParams")();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }

  np::ndarray QFunction(const np::ndarray &state) {
    try {
      return bp::extract<np::ndarray>(learner.attr("QFunction")(state));
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }
};

TFLearner::TFLearner(const NetworkSpec &spec) {
  impl = make_unique<TFLearnerImpl>(spec);
}

TFLearner::~TFLearner() = default;

void TFLearner::Learn(const QLearnBatch &batch) { impl->Learn(batch); }

void TFLearner::UpdateTargetParams(void) { impl->UpdateTargetParams(); }

np::ndarray TFLearner::QFunction(const np::ndarray &state) {
  return impl->QFunction(state);
}
