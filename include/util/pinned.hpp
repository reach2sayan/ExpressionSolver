#pragma once

// Two empty bases for the classes that have to suppress a special member, so
// that the intent is a name rather than four `= delete` lines to be read and
// compared:
//
//   class scoped_value : private pinned { ... };      // stays where it is
//   class Compiler : private noncopyable { ... };     // one owner, but movable
//
// Inherit privately.  Both are empty, so a derived class pays nothing for them.
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
