
#pragma once

namespace learning {

static constexpr unsigned MOMENTS_BATCH_SIZE = 100;
static constexpr float REWARD_DELAY_DISCOUNT = 0.9f;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 100000;

static constexpr float INITIAL_TEMPERATURE = 1.0f;
static constexpr float TARGET_TEMPERATURE = 0.1f;

static constexpr float INITIAL_LEARN_RATE = 0.001f;
static constexpr float TARGET_LEARN_RATE = 0.00001f;
}
