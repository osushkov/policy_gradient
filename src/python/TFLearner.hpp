#pragma once

#include "../util/Common.hpp"
#include "NetworkSpec.hpp"
#include <boost/python/numpy.hpp>
#include <cassert>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

struct PolicyLearnBatch {
  PolicyLearnBatch()
      : initialStates(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        actionsTaken(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<int>())),
        rewardsGained(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<float>())),
        learnRate(1.0f) {}

  PolicyLearnBatch(const np::ndarray &initialStates,
                   const np::ndarray &actionsTaken,
                   const np::ndarray &rewardsGained,
                   float learnRate)
      : initialStates(initialStates), actionsTaken(actionsTaken),
        rewardsGained(rewardsGained), learnRate(learnRate) {
    assert(learnRate > 0.0f);
  }

  np::ndarray initialStates;
  np::ndarray actionsTaken;  // action indices
  np::ndarray rewardsGained; // floats
  float learnRate;
};

struct ValueLearnBatch {
  ValueLearnBatch()
      : states(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        rewardsGained(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<float>())),
        learnRate(1.0f) {}

  ValueLearnBatch(const np::ndarray &states,
                  const np::ndarray &rewardsGained,
                  float learnRate)
      : states(states), rewardsGained(rewardsGained), learnRate(learnRate) {
    assert(learnRate > 0.0f);
  }

  np::ndarray states;
  np::ndarray rewardsGained;
  float learnRate;
};

class TFLearner {
public:
  TFLearner(const NetworkSpec &spec);
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void Learn(const PolicyLearnBatch &batch);
  void LearnValue(const ValueLearnBatch &batch);
  np::ndarray PolicyFunction(const np::ndarray &state);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
}
