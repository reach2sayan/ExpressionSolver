#pragma once

#include "rt/archive/codec.hpp"
#include "util/error.hpp"
#include "util/export.hpp"

#include <boost/describe.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// A ddx file: an eight-byte tag, a prologue, and one
// checksummed payload.
namespace ddx::rt::detail {

// Little-endian and fixed-width throughout, so a file is portable; size_t is
// the one host type it leans on.
static_assert(sizeof(std::size_t) == 8);

enum class ScalarKind : std::uint8_t { Integral, Floating };

struct FileHeader {
  // The tag, not a value: eight raw bytes both sides handle before the
  // described walk, and so deliberately outside it.
  std::string_view magic;
  std::uint32_t format = 0;
  std::uint32_t schema = 0;
  std::uint8_t scalar_size = 0;
  ScalarKind scalar_kind = ScalarKind::Integral;
  std::uint32_t model_nodes = 0;
  std::uint64_t model_digest = 0;
  std::uint32_t payload_crc = 0;
};

// Two fields are deliberately absent.
// An opcode count: the label table is length-prefixed under the checksum,
// A payload length: the checksum already covers every byte after the prologue,
BOOST_DESCRIBE_STRUCT(FileHeader, (),
                      (format, schema, scalar_size, scalar_kind, model_nodes,
                       model_digest, payload_crc))

struct Container {
  // What a caller slicing a file has to know, and the only sizes anyone does.
  static constexpr std::size_t magic_bytes = 8;
  static constexpr std::size_t header_bytes = 40;
  static constexpr std::size_t reserved_bytes =
      header_bytes - magic_bytes - wire_bytes<FileHeader>();
  static_assert(reserved_bytes == 6);

  // `payload_crc` is computed here
  [[nodiscard]] static DDX_API std::vector<std::byte>
  pack(std::string_view magic, const FileHeader &h,
       std::span<const std::byte> payload);

  // Magic, the zero reserved tail, and the checksum
  [[nodiscard]] static DDX_API
      result<std::pair<FileHeader, std::span<const std::byte>>>
      unpack(std::span<const std::byte> bytes, std::string_view magic);

  [[nodiscard]] static DDX_API result<std::vector<std::byte>>
  read(const std::filesystem::path &path);
  // Staged beside the target and renamed over it: a reader never sees a
  // half-written file, and a failed save leaves the old one standing.
  [[nodiscard]] static DDX_API result<void>
  write(const std::filesystem::path &path, std::span<const std::byte> bytes);

  // Payloads with a described shape, encoded end to end into one buffer,
  // reserved exactly from the Counter's pass.
  template <typename... Vs>
  [[nodiscard]] static std::vector<std::byte> encode(const Vs &...vs) {
    std::vector<std::byte> out;
    out.reserve(wire_size(vs...));
    Writer w{out};
    (wire(w, std::as_const(vs)), ...);
    return out;
  }

  // Whole or not at all: trailing bytes are as much a corrupt file as short
  // ones.  No opcode remap, so a payload naming one is a caller for load().
  template <typename V>
  [[nodiscard]] static result<V> decode(std::span<const std::byte> bytes) {
    V v{};
    Reader r{bytes, {}};
    wire(r, v);
    if (!r.ok() || !r.spent()) {
      return fail(errc::archive_corrupt);
    }
    return v;
  }

  // The payload's shape and the prologue's, folded together: a field added to
  // either refuses old files without a constant being bumped by hand.
  template <typename V> [[nodiscard]] static constexpr std::uint32_t stamp() {
    return schema_of<FileHeader, V>();
  }

private:
  static DDX_API void put_header(const FileHeader &h,
                                 std::span<std::byte> into);
  static DDX_API result<FileHeader> get_header(std::span<const std::byte> bytes,
                                               std::string_view magic);
  static DDX_API std::uint32_t checksum(std::span<const std::byte> bytes);
};

} // namespace ddx::rt::detail
