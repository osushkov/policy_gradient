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

  virtual void Learn(const PolicyLearnBatch &batch) = 0;
  virtual void LearnValue(const ValueLearnBatch &batch) = 0;
  virtual np::ndarray PolicyFunction(const np::ndarray &state) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void Learn(const PolicyLearnBatch &batch) override {
    get_override("Learn")(batch);
  }

  void LearnValue(const ValueLearnBatch &batch) override {
    get_override("LearnValue")(batch);
  }

  np::ndarray PolicyFunction(const np::ndarray &state) override {
    return get_override("PolicyFunction")(state);
  }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();

  bp::class_<NetworkSpec>("NetworkSpec")
      .def_readonly("numInputs", &NetworkSpec::numInputs)
      .def_readonly("numOutputs", &NetworkSpec::numOutputs)
      .def_readonly("maxBatchSize", &NetworkSpec::maxBatchSize);

  bp::class_<PolicyLearnBatch>("PolicyLearnBatch")
      .def_readonly("initialStates", &PolicyLearnBatch::initialStates)
      .def_readonly("actionsTaken", &PolicyLearnBatch::actionsTaken)
      .def_readonly("rewardsGained", &PolicyLearnBatch::rewardsGained)
      .def_readonly("learnRate", &PolicyLearnBatch::learnRate);

  bp::class_<ValueLearnBatch>("ValueLearnBatch")
      .def_readonly("states", &ValueLearnBatch::states)
      .def_readonly("rewardsGained", &ValueLearnBatch::rewardsGained)
      .def_readonly("learnRate", &ValueLearnBatch::learnRate);

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

  void Learn(const PolicyLearnBatch &batch) {
    try {
      learner.attr("Learn")(batch);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }

  void LearnValue(const ValueLearnBatch &batch) {
    try {
      learner.attr("LearnValue")(batch);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << python::ParseException() << std::endl;
      throw e;
    }
  }

  np::ndarray PolicyFunction(const np::ndarray &state) {
    try {
      return bp::extract<np::ndarray>(learner.attr("PolicyFunction")(state));
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

void TFLearner::Learn(const PolicyLearnBatch &batch) { impl->Learn(batch); }

void TFLearner::LearnValue(const ValueLearnBatch &batch) {
  impl->LearnValue(batch);
}

np::ndarray TFLearner::PolicyFunction(const np::ndarray &state) {
  return impl->PolicyFunction(state);
}
