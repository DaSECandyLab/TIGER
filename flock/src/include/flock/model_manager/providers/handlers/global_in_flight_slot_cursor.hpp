#pragma once

#include <cstddef>
#include <optional>

namespace flock {

// Keeps the existing filesystem-slot coordination semantics while avoiding a
// full scan from slot zero for every request.
class GlobalInFlightSlotCursor {
public:
    explicit GlobalInFlightSlotCursor(size_t slot_count)
        : slot_count_(slot_count) {
    }

    template<typename ClaimFn>
    std::optional<size_t> TryClaim(ClaimFn&& try_claim) {
        if (slot_count_ == 0) {
            return std::nullopt;
        }

        auto slot_idx = next_slot_;
        for (size_t scanned = 0; scanned < slot_count_; ++scanned) {
            if (try_claim(slot_idx)) {
                next_slot_ = Increment(slot_idx);
                return slot_idx;
            }
            slot_idx = Increment(slot_idx);
        }
        return std::nullopt;
    }

private:
    size_t Increment(size_t slot_idx) const {
        ++slot_idx;
        return slot_idx == slot_count_ ? 0 : slot_idx;
    }

    size_t slot_count_;
    size_t next_slot_ = 0;
};

}// namespace flock
