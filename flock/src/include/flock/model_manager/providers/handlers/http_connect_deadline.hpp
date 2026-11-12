#pragma once

#include <stdexcept>
#include <string>

namespace flock {

// Zero means no explicit override: preserve libcurl's existing default.
// This execution safeguard is not a batching or mechanism tuning parameter.
inline long ParseHttpConnectDeadlineSeconds(const char* value) {
    if (value == nullptr) {
        return 0;
    }
    const std::string text(value);
    size_t consumed = 0;
    const long seconds = std::stol(text, &consumed);
    if (consumed != text.size() || seconds <= 0) {
        throw std::invalid_argument("FLOCK_HTTP_CONNECT_TIMEOUT_SECONDS must be a positive integer");
    }
    return seconds;
}

} // namespace flock
