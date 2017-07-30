
#include "LearningAgent.hpp"
#include "../python/NetworkSpec.hpp"
#include "../python/PythonContext.hpp"
#include "../python/PythonUtil.hpp"
#include "../python/TFLearner.hpp"
#include "../util/Common.hpp"
#include "../util/Math.hpp"
#include "../util/Timer.hpp"
#include "Constants.hpp"
#include "TrainingSample.hpp"

#include <cassert>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  float pRandom;
  float temperature;

  python::PythonThreadContext ptctx;
  uptr<python::TFLearner> learner;

  unsigned itersSinceTargetUpdated;

  LearningAgentImpl()
      : pRandom(0.0f), temperature(0.0001f), ptctx(python::GlobalContext()) {
    python::PythonContextLock pl(ptctx);

    python::NetworkSpec spec(BOARD_WIDTH * BOARD_HEIGHT * 2,
                             GameAction::ALL_ACTIONS().size(),
                             MOMENTS_BATCH_SIZE);

    learner = make_unique<python::TFLearner>(spec);
    itersSinceTargetUpdated = TARGET_FUNCTION_UPDATE_RATE + 1;
  }

  GameAction SelectAction(const GameState *state) {
    assert(state != nullptr);
    auto r = chooseBestAction(*state, LearningAgent::EncodeGameState(state));
    // std::cout << "sa: " << r << std::endl;
    return r;
  }

  vector<GameAction>
  SelectLearningActions(const vector<pair<GameState *, EVector>> &states) {
    vector<GameAction> actions = chooseBestActions(states);
    for (unsigned i = 0; i < actions.size(); i++) {
      if (util::RandInterval(0.0, 1.0) < pRandom) {
        actions[i] = chooseExplorativeAction(*states[i].first);
      }

      // std::cout << "sla: " << actions[i] << std::endl;
    }
    return actions;
  }

  void Learn(const vector<ExperienceMoment> &moments, float learnRate) {
    if (itersSinceTargetUpdated > TARGET_FUNCTION_UPDATE_RATE) {
      learner->UpdateTargetParams();
      itersSinceTargetUpdated = 0;
    }

    learner->Learn(makeQBatch(moments, learnRate));
    itersSinceTargetUpdated++;
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

  // TODO: probably dont need this.
  float GetQValue(const GameState &state, const GameAction &action) {
    EMatrix qvalues = learnerInference(LearningAgent::EncodeGameState(&state));
    return qvalues(GameAction::ACTION_INDEX(action), 0);
  }

  python::QLearnBatch makeQBatch(const vector<ExperienceMoment> &moments,
                                 float learnRate) {
    EMatrix initialStates(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    EMatrix successorStates(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    std::vector<int> actionsTaken(moments.size());
    std::vector<bool> isEndStateTerminal(moments.size());
    std::vector<float> rewardsGained(moments.size());

    for (unsigned i = 0; i < moments.size(); i++) {
      initialStates.row(i) = moments[i].initialState;
      successorStates.row(i) = moments[i].successorState;
      actionsTaken[i] = GameAction::ACTION_INDEX(moments[i].actionTaken);
      isEndStateTerminal[i] = moments[i].isSuccessorTerminal;
      rewardsGained[i] = moments[i].reward;
    }

    return python::QLearnBatch(
        python::ToNumpy(initialStates), python::ToNumpy(successorStates),
        python::ToNumpy(actionsTaken), python::ToNumpy(isEndStateTerminal),
        python::ToNumpy(rewardsGained), REWARD_DELAY_DISCOUNT, learnRate);
  }

  GameAction chooseBestAction(const GameState &state,
                              const EVector &encodedState) {
    EMatrix qvalues = learnerInference(encodedState);
    std::vector<unsigned> availableActions = state.AvailableActions();
    assert(availableActions.size() > 0);

    GameAction bestAction = GameAction::ACTION(availableActions[0]);
    float bestQValue = qvalues(0, availableActions[0]);

    for (unsigned i = 1; i < availableActions.size(); i++) {
      float qv = qvalues(0, availableActions[i]);
      if (qv > bestQValue) {
        bestQValue = qv;
        bestAction = GameAction::ACTION(availableActions[i]);
      }
    }

    return bestAction;
  }

  vector<GameAction>
  chooseBestActions(const vector<pair<GameState *, EVector>> &states) {
    assert(states.size() <= MOMENTS_BATCH_SIZE);

    EMatrix encodedStates(states.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    for (unsigned i = 0; i < states.size(); i++) {
      encodedStates.row(i) = states[i].second;
    }

    EMatrix qvalues = learnerInferenceBatch(encodedStates);

    vector<GameAction> result;
    for (unsigned i = 0; i < states.size(); i++) {
      std::vector<unsigned> availableActions =
          states[i].first->AvailableActions();
      assert(availableActions.size() > 0);

      GameAction bestAction = GameAction::ACTION(availableActions[0]);
      float bestQValue = qvalues(i, availableActions[0]);

      for (unsigned j = 1; j < availableActions.size(); j++) {
        float qv = qvalues(i, availableActions[j]);
        if (qv > bestQValue) {
          bestQValue = qv;
          bestAction = GameAction::ACTION(availableActions[j]);
        }
      }

      result.emplace_back(bestAction);
    }

    return result;
  }

  GameAction chooseExplorativeAction(const GameState &state) {
    auto aa = state.AvailableActions();
    return GameAction::ACTION(aa[rand() % aa.size()]);
  }

  EMatrix learnerInference(const EVector &encodedState) {
    EMatrix qvalues =
        python::ToEigen2D(learner->QFunction(python::ToNumpy(encodedState)));
    assert(qvalues.cols() ==
           static_cast<int>(GameAction::ALL_ACTIONS().size()));
    assert(qvalues.rows() == 1);

    return qvalues;
  }

  EMatrix learnerInferenceBatch(const EMatrix &encodedStates) {
    EMatrix qvalues =
        python::ToEigen2D(learner->QFunction(python::ToNumpy(encodedStates)));
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

void LearningAgent::SetPRandom(float pRandom) { impl->pRandom = pRandom; }

void LearningAgent::SetTemperature(float temperature) {
  impl->temperature = temperature;
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

float LearningAgent::GetQValue(const GameState &state,
                               const GameAction &action) {
  python::PythonContextLock pl(impl->ptctx);
  return impl->GetQValue(state, action);
}
