
#include "Trainer.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/Constants.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"
#include "util/Common.hpp"
#include "util/Timer.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>
#include <vector>

using namespace learning;

struct PlayoutAgent {
  LearningAgent *agent;
  ExperienceMemory *memory;

  vector<EVector> stateHistory;
  vector<GameAction> actionHistory;

  PlayoutAgent(LearningAgent *agent, ExperienceMemory *memory)
      : agent(agent), memory(memory) {}

  bool havePreviousState(void) { return stateHistory.size() > 0; }

  void addTransitionToMemory(EVector &curState, float reward, bool isTerminal) {
    if (!havePreviousState()) {
      return;
    }

    EVector &prevState = stateHistory[stateHistory.size() - 1];
    GameAction &performedAction = actionHistory[actionHistory.size() - 1];

    memory->AddExperience(ExperienceMoment(prevState, performedAction, curState,
                                           reward, isTerminal));
  }

  void addMoveToHistory(const EVector &state, const GameAction &action) {
    stateHistory.push_back(state);
    actionHistory.push_back(action);
  }
};

struct Trainer::TrainerImpl {
  vector<ProgressCallback> callbacks;
  atomic<unsigned> numLearnIters;

  void AddProgressCallback(ProgressCallback callback) {
    callbacks.push_back(callback);
  }

  uptr<LearningAgent> TrainAgent(unsigned iters) {
    auto experienceMemory =
        make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);

    uptr<LearningAgent> agent = make_unique<LearningAgent>();
    trainAgent(agent.get(), experienceMemory.get(), iters, INITIAL_PRANDOM,
               TARGET_PRANDOM);

    return move(agent);
  }

  void trainAgent(LearningAgent *agent, ExperienceMemory *memory,
                  unsigned iters, float initialPRandom, float targetPRandom) {

    numLearnIters = 0;

    std::thread playoutThread =
        startPlayoutThread(agent, memory, iters, initialPRandom, targetPRandom);

    std::thread learnThread = startLearnThread(agent, memory, iters);

    playoutThread.join();
    learnThread.join();
  }

  std::thread startPlayoutThread(LearningAgent *agent, ExperienceMemory *memory,
                                 unsigned iters, float initialPRandom,
                                 float targetPRandom) {

    return std::thread(
        [this, agent, memory, iters, initialPRandom, targetPRandom]() {
          float pRandDecay = powf(targetPRandom / initialPRandom, 1.0f / iters);
          assert(pRandDecay > 0.0f && pRandDecay <= 1.0f);

          // float tempDecay = powf(TARGET_TEMPERATURE / INITIAL_TEMPERATURE,
          // 1.0f / iters); assert(tempDecay > 0.0f && tempDecay <= 1.0f);

          while (true) {
            unsigned doneIters = numLearnIters.load();
            if (doneIters >= iters) {
              break;
            }

            float prand = initialPRandom * powf(pRandDecay, doneIters);
            // float temp = INITIAL_TEMPERATURE * powf(tempDecay, doneIters);
            agent->SetPRandom(prand);
            // agent->SetTemperature(temp);

            this->playoutRoundVsSelf(agent, memory);
            // this->playoutRoundVsRandom(agent, memory);
          }
        });
  }

  std::thread startLearnThread(LearningAgent *agent, ExperienceMemory *memory,
                               unsigned iters) {
    return std::thread([this, agent, memory, iters]() {
      while (memory->NumMemories() < 10 * MOMENTS_BATCH_SIZE) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      float learnRateDecay =
          powf(TARGET_LEARN_RATE / INITIAL_LEARN_RATE, 1.0f / iters);
      assert(learnRateDecay > 0.0f && learnRateDecay < 1.0f);

      for (unsigned i = 0; i < iters; i++) {
        for (auto &cb : this->callbacks) {
          cb(agent, i);
        }

        float learnRate = INITIAL_LEARN_RATE * powf(learnRateDecay, i);
        agent->Learn(memory->Sample(MOMENTS_BATCH_SIZE), learnRate);
        this->numLearnIters++;
      }

      agent->Finalise();
    });
  }

  void playoutRoundVsSelf(LearningAgent *agent, ExperienceMemory *memory) {
    GameRules *rules = GameRules::Instance();

    GameState initialState = rules->InitialState();
    EVector encodedInitialState = LearningAgent::EncodeGameState(&initialState);

    vector<GameState> curStates;
    std::vector<PlayoutAgent> playoutAgents;
    for (unsigned i = 0; i < MOMENTS_BATCH_SIZE; i++) {
      curStates.emplace_back(generateStartState());

      playoutAgents.emplace_back(agent, memory);
      playoutAgents.emplace_back(agent, memory);
    }

    vector<bool> stateActive(curStates.size(), true);

    unsigned curPlayerIndex = 0;
    while (true) {
      vector<pair<GameState *, EVector>> encodedStates;
      for (unsigned i = 0; i < curStates.size(); i++) {
        if (stateActive[i]) {
          encodedStates.emplace_back(
              &curStates[i], LearningAgent::EncodeGameState(&curStates[i]));
        } else {
          encodedStates.emplace_back(&initialState, encodedInitialState);
        }
      }

      vector<GameAction> actions = agent->SelectLearningActions(encodedStates);

      unsigned numActiveStates = 0;
      for (unsigned i = 0; i < curStates.size(); i++) {
        if (!stateActive[i]) {
          continue;
        }
        numActiveStates++;

        EVector encodedState = encodedStates[i].second;

        PlayoutAgent &curPlayer = playoutAgents[i * 2 + curPlayerIndex];
        PlayoutAgent &otherPlayer =
            playoutAgents[i * 2 + (curPlayerIndex + 1) % 2];

        curPlayer.addTransitionToMemory(encodedState, 0.0f, false);
        curPlayer.addMoveToHistory(encodedState, actions[i]);

        curStates[i] = curStates[i].SuccessorState(actions[i]);

        switch (rules->GameCompletionState(curStates[i])) {
        case CompletionState::WIN:
          encodedState = LearningAgent::EncodeGameState(&curStates[i]);
          curPlayer.addTransitionToMemory(encodedState, 1.0f, true);
          otherPlayer.addTransitionToMemory(encodedState, -1.0f, true);
          stateActive[i] = false;
          break;
        case CompletionState::LOSS:
          assert(false); // This actually shouldn't be possible.
          break;
        case CompletionState::DRAW:
          encodedState = LearningAgent::EncodeGameState(&curStates[i]);
          curPlayer.addTransitionToMemory(encodedState, 0.0f, true);
          otherPlayer.addTransitionToMemory(encodedState, 0.0f, true);
          stateActive[i] = false;
          break;
        case CompletionState::UNFINISHED:
          curStates[i].FlipState();
          break;
        }
      }

      if (numActiveStates == 0) {
        return;
      }
      curPlayerIndex = (curPlayerIndex + 1) % 2;
    }
  }

  void playoutRoundVsRandom(LearningAgent *agent, ExperienceMemory *memory) {
    GameRules *rules = GameRules::Instance();

    vector<PlayoutAgent> playoutAgents;
    RandomAgent randomAgent;

    GameState initialState = rules->InitialState();
    EVector encodedInitialState = LearningAgent::EncodeGameState(&initialState);

    vector<GameState> curStates;
    for (unsigned i = 0; i < MOMENTS_BATCH_SIZE; i++) {
      playoutAgents.emplace_back(agent, memory);
      curStates.emplace_back(generateStartState());
    }

    vector<bool> stateActive(curStates.size(), true);

    unsigned curPlayerIndex = rand() % 2;
    while (true) {
      vector<pair<GameState *, EVector>> encodedStates;
      for (unsigned i = 0; i < curStates.size(); i++) {
        if (stateActive[i]) {
          encodedStates.emplace_back(
              &curStates[i], LearningAgent::EncodeGameState(&curStates[i]));
        } else {
          encodedStates.emplace_back(&initialState, encodedInitialState);
        }
      }

      vector<GameAction> actions;
      if (curPlayerIndex == 0) {
        actions = agent->SelectLearningActions(encodedStates);
      } else {
        for (unsigned i = 0; i < curStates.size(); i++) {
          if (stateActive[i]) {
            actions.push_back(randomAgent.SelectAction(&curStates[i]));
          } else {
            actions.push_back(GameAction::ACTION(0));
          }
        }
      }

      unsigned numActiveStates = 0;
      for (unsigned i = 0; i < curStates.size(); i++) {
        if (!stateActive[i]) {
          continue;
        }
        numActiveStates++;

        EVector encodedState = encodedStates[i].second;

        if (curPlayerIndex == 0) {
          playoutAgents[i].addTransitionToMemory(encodedState, 0.0f, false);
          playoutAgents[i].addMoveToHistory(encodedState, actions[i]);
        }

        curStates[i] = curStates[i].SuccessorState(actions[i]);

        switch (rules->GameCompletionState(curStates[i])) {
        case CompletionState::WIN:
          encodedState = LearningAgent::EncodeGameState(&curStates[i]);
          playoutAgents[i].addTransitionToMemory(
              encodedState, curPlayerIndex == 0 ? 1.0f : -1.0f, true);
          stateActive[i] = false;
          break;
        case CompletionState::LOSS:
          assert(false); // This actually shouldn't be possible.
          break;
        case CompletionState::DRAW:
          encodedState = LearningAgent::EncodeGameState(&curStates[i]);
          playoutAgents[i].addTransitionToMemory(encodedState, 0.0f, true);
          stateActive[i] = false;
          break;
        case CompletionState::UNFINISHED:
          curStates[i].FlipState();
          break;
        }
      }

      if (numActiveStates == 0) {
        return;
      }
      curPlayerIndex = (curPlayerIndex + 1) % 2;
    }
  }

  GameState generateStartState(void) {
    GameRules *rules = GameRules::Instance();

    RandomAgent agent;
    std::vector<GameState> states;

    GameState curState(rules->InitialState());
    bool isFinished = false;

    while (!isFinished) {
      states.push_back(curState);
      GameAction action = agent.SelectAction(&curState);
      curState = curState.SuccessorState(action);

      switch (rules->GameCompletionState(curState)) {
      case CompletionState::WIN:
      case CompletionState::LOSS:
      case CompletionState::DRAW:
        isFinished = true;
        break;
      case CompletionState::UNFINISHED:
        curState.FlipState();
        break;
      }
    }

    unsigned backtrack = 4;
    if (states.size() <= backtrack) {
      return states[0];
    } else {
      return states[rand() % (states.size() - backtrack)];
    }
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

void Trainer::AddProgressCallback(ProgressCallback callback) {
  impl->AddProgressCallback(callback);
}

uptr<LearningAgent> Trainer::TrainAgent(unsigned iters) {
  return impl->TrainAgent(iters);
}
