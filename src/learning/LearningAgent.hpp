
#pragma once

#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include "../util/Common.hpp"
#include "Agent.hpp"
#include "ExperienceMoment.hpp"

#include <iostream>
#include <utility>
#include <vector>

using namespace connectfour;

namespace learning {

class LearningAgent : public Agent {
public:
  static EVector EncodeGameState(const GameState *state);

  LearningAgent();
  virtual ~LearningAgent();

  static uptr<LearningAgent> Read(std::istream &in);
  void Write(std::ostream &out);

  GameAction SelectAction(const GameState *state) override;

  void SetTemperature(float temperature);

  vector<GameAction>
  SelectLearningActions(const vector<pair<GameState *, EVector>> &states);

  void Learn(const vector<ExperienceMoment> &moments, float learnRate);

  void Finalise(void);

private:
  struct LearningAgentImpl;
  uptr<LearningAgentImpl> impl;
};
}
