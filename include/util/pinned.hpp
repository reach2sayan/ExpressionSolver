#pragma once

namespace ddx::impl {

namespace pinned_ {

// One owner, and the ownership can be handed on.
class noncopyable {
protected:
  constexpr noncopyable() = default;
  ~noncopyable() = default;
  noncopyable(noncopyable &&) = default;
  noncopyable &operator=(noncopyable &&) = default;

public:
  noncopyable(const noncopyable &) = delete;
  noncopyable &operator=(const noncopyable &) = delete;
};

// The same, and the ownership cannot be handed on either:
class pinned : private noncopyable {
protected:
  constexpr pinned() = default;
  ~pinned() = default;

public:
  pinned(pinned &&) = delete;
  pinned &operator=(pinned &&) = delete;
};

} // namespace pinned_

using pinned_::noncopyable;
using pinned_::pinned;

} // namespace ddx::impl
