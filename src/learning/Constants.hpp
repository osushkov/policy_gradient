
#pragma once

namespace learning {

static constexpr unsigned MOMENTS_BATCH_SIZE = 500;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 5000;
static constexpr float REWARD_DELAY_DISCOUNT = 0.9f;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 100000;

static constexpr float INITIAL_PRANDOM = 0.5f;
static constexpr float TARGET_PRANDOM = 0.05f;

static constexpr float INITIAL_LEARN_RATE = 0.001f;
static constexpr float TARGET_LEARN_RATE = 0.00001f;
}
