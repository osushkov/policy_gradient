
#include "LearningAgent.hpp"
#include "../python/NetworkSpec.hpp"
#include "../python/PythonContext.hpp"
#include "../python/PythonUtil.hpp"
#include "../python/TFLearner.hpp"
#include "../util/Common.hpp"
#include "../util/Math.hpp"
#include "../util/Timer.hpp"
#include "Constants.hpp"

#include <cassert>

using namespace learning;
using namespace std;

struct LearningAgent::LearningAgentImpl {
  float exploration;

  python::PythonThreadContext ptctx;
  uptr<python::TFLearner> learner;

  LearningAgentImpl() : exploration(0.0f), ptctx(python::GlobalContext()) {
    python::PythonContextLock pl(ptctx);

    python::NetworkSpec spec(BOARD_WIDTH * BOARD_HEIGHT * 2,
                             GameAction::ALL_ACTIONS().size(),
                             MOMENTS_BATCH_SIZE);

    learner = make_unique<python::TFLearner>(spec);
  }

  GameAction SelectAction(const GameState *state) {
    assert(state != nullptr);
    return sampleAction(*state, LearningAgent::EncodeGameState(state));
  }

  vector<GameAction>
  SelectLearningActions(const vector<pair<GameState *, EVector>> &states) {
    auto actions = sampleActions(states);
    for (unsigned i = 0; i < actions.size(); i++) {
      if (util::RandInterval(0.0, 1.0) < exploration) {
        actions[i] = chooseExplorativeAction(*states[i].first);
      }
    }
    return actions;
  }

  void Learn(const vector<ExperienceMoment> &moments, float learnRate) {
    learner->Learn(makePolicyBatch(moments, learnRate));
    learner->LearnValue(makeValueBatch(moments, learnRate));
  }

  void Finalise(void) {
    // if (learningNet == nullptr) {
    //   return;
    // }
    //
    // targetNet = learningNet->RefreshAndGetTarget();
    // learningNet.release();
    // learningNet = nullptr;
  }

  python::PolicyLearnBatch
  makePolicyBatch(const vector<ExperienceMoment> &moments, float learnRate) {
    EMatrix initialStates(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    vector<int> actionsTaken(moments.size());
    vector<float> rewardsGained(moments.size());

    for (unsigned i = 0; i < moments.size(); i++) {
      initialStates.row(i) = moments[i].initialState;
      actionsTaken[i] = GameAction::ACTION_INDEX(moments[i].actionTaken);
      rewardsGained[i] = moments[i].futureRewards;
    }

    return python::PolicyLearnBatch(python::ToNumpy(initialStates),
                                    python::ToNumpy(actionsTaken),
                                    python::ToNumpy(rewardsGained),
                                    learnRate);
  }

  python::ValueLearnBatch
  makeValueBatch(const vector<ExperienceMoment> &moments, float learnRate) {
    EMatrix states(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    vector<float> rewardsGained(moments.size());

    for (unsigned i = 0; i < moments.size(); i++) {
      states.row(i) = moments[i].initialState;
      rewardsGained[i] = moments[i].futureRewards;
    }

    return python::ValueLearnBatch(python::ToNumpy(states),
                                   python::ToNumpy(rewardsGained),
                                   learnRate);
  }

  GameAction sampleAction(const GameState &state, const EVector &encodedState) {
    EMatrix policyValues = learnerInference(encodedState);
    vector<unsigned> availableActions = state.AvailableActions();
    assert(availableActions.size() > 0);

    vector<float> actionWeights;
    for (unsigned i = 0; i < availableActions.size(); i++) {
      actionWeights.emplace_back(policyValues(0, availableActions[i]));
    }

    // unsigned bestIndex = 0;
    // float bestWeight = actionWeights[0];
    // for (unsigned i = 1; i < actionWeights.size(); i++) {
    //   if (actionWeights[i] > bestWeight) {
    //     bestWeight = actionWeights[i];
    //     bestIndex = i;
    //   }
    // }

    unsigned bestIndex = util::SoftmaxSample(actionWeights, 0.1);
    return GameAction::ACTION(availableActions[bestIndex]);
  }

  vector<GameAction> sampleActions(const vector<pair<GameState *, EVector>> &states) {
    assert(states.size() <= MOMENTS_BATCH_SIZE);

    EMatrix encodedStates(states.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    for (unsigned i = 0; i < states.size(); i++) {
      encodedStates.row(i) = states[i].second;
    }

    EMatrix policyValues = learnerInferenceBatch(encodedStates);

    vector<GameAction> result;
    for (unsigned i = 0; i < states.size(); i++) {
      std::vector<unsigned> availableActions =
          states[i].first->AvailableActions();
      assert(availableActions.size() > 0);

      vector<float> actionWeights;
      for (unsigned i = 0; i < availableActions.size(); i++) {
        actionWeights.emplace_back(policyValues(0, availableActions[i]));
      }

      unsigned sampledIndex = util::SoftmaxSample(actionWeights, 1.0);
      result.emplace_back(GameAction::ACTION(availableActions[sampledIndex]));
    }

    return result;
  }

  GameAction chooseExplorativeAction(const GameState &state) {
    auto aa = state.AvailableActions();
    return GameAction::ACTION(aa[rand() % aa.size()]);
  }

  EMatrix learnerInference(const EVector &encodedState) {
    EMatrix qvalues = python::ToEigen2D(
        learner->PolicyFunction(python::ToNumpy(encodedState)));
    assert(qvalues.cols() ==
           static_cast<int>(GameAction::ALL_ACTIONS().size()));
    assert(qvalues.rows() == 1);
    return qvalues;
  }

  EMatrix learnerInferenceBatch(const EMatrix &encodedStates) {
    EMatrix qvalues = python::ToEigen2D(
        learner->PolicyFunction(python::ToNumpy(encodedStates)));
    assert(qvalues.cols() ==
           static_cast<int>(GameAction::ALL_ACTIONS().size()));
    assert(qvalues.rows() == encodedStates.rows());
    return qvalues;
  }
};

EVector LearningAgent::EncodeGameState(const GameState *state) {
  EVector result(2 * BOARD_WIDTH * BOARD_HEIGHT);
  result.fill(0.0f);

  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {
      unsigned ri = 2 * (c + r * BOARD_WIDTH);

      switch (state->GetCell(r, c)) {
      case CellState::MY_TOKEN:
        result(ri) = 1.0f;
        break;
      case CellState::OPPONENT_TOKEN:
        result(ri + 1) = 1.0f;
        break;
      default:
        break;
      }

      ri++;
    }
  }

  return result;
}

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

uptr<LearningAgent> LearningAgent::Read(std::istream &in) {
  uptr<LearningAgent> result = make_unique<LearningAgent>();
  // result->impl->targetNet = neuralnetwork::Network::Read(in);
  // result->impl->learningNet.release();
  // result->impl->learningNet = nullptr;
  return result;
}

void LearningAgent::Write(std::ostream &out) {
  python::PythonContextLock pl(impl->ptctx);
  /*impl->targetNet->Write(out);*/
}

GameAction LearningAgent::SelectAction(const GameState *state) {
  python::PythonContextLock pl(impl->ptctx);
  return impl->SelectAction(state);
}

void LearningAgent::SetExploration(float exploration) {
  impl->exploration = exploration;
}

vector<GameAction> LearningAgent::SelectLearningActions(
    const vector<pair<GameState *, EVector>> &states) {

  python::PythonContextLock pl(impl->ptctx);
  return impl->SelectLearningActions(states);
}

void LearningAgent::Learn(const vector<ExperienceMoment> &moments,
                          float learnRate) {
  python::PythonContextLock pl(impl->ptctx);
  impl->Learn(moments, learnRate);
}

void LearningAgent::Finalise(void) {
  python::PythonContextLock pl(impl->ptctx);
  impl->Finalise();
}
