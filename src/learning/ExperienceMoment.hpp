#pragma once

#include "../connectfour/GameState.hpp"
#include "../util/Math.hpp"
#include <cstdlib>

using namespace connectfour;

namespace learning {

struct ExperienceMoment {
  EVector initialState;
  GameAction actionTaken;
  float futureRewards;

  ExperienceMoment() = default;
  ExperienceMoment(EVector initialState, GameAction actionTaken,
                   float futureRewards)
      : initialState(initialState), actionTaken(actionTaken),
        futureRewards(futureRewards) {}
};
}
