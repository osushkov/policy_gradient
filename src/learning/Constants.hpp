
#pragma once

namespace learning {

static constexpr unsigned MOMENTS_BATCH_SIZE = 100;
static constexpr float REWARD_DELAY_DISCOUNT = 0.9f;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 10000;

static constexpr float INITIAL_EXPLORATION = 0.5f;
static constexpr float TARGET_EXPLORATION = 0.01f;

static constexpr float INITIAL_LEARN_RATE = 0.0001f;
static constexpr float TARGET_LEARN_RATE = 0.000001f;
}
