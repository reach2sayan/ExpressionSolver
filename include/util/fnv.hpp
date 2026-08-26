#pragma once

#include <cstdint>
#include <string_view>

namespace ddx::impl {

// FNV-1a, 64-bit.  Not a digest: a fold that gives one number to a byte stream
// nothing reads back, so the only property asked of it is that two streams
// which differ anywhere are unlikely to land on the same number.  Shared so
// that the archive digest, the JIT cache key and the bit-exactness gate are
// one hash rather than three that drift.
inline constexpr std::uint64_t fnv64_basis = 14695981039346656037ULL;
inline constexpr std::uint64_t fnv64_prime = 1099511628211ULL;

// A word as its eight bytes, low first: a fixed width, so a field is never
// confusable with a shorter one that happens to share its value.
constexpr void fold64(std::uint64_t &h, std::uint64_t w) noexcept {
  for (int i = 0; i < 8; ++i) {
    h = (h ^ ((w >> (i * 8)) & 0xFF)) * fnv64_prime;
  }
}

constexpr void fold_bytes(std::uint64_t &h, std::string_view bytes) noexcept {
  for (const char c : bytes) {
    h = (h ^ static_cast<unsigned char>(c)) * fnv64_prime;
  }
}

} // namespace ddx::impl
