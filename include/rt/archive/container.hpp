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

struct FileHeader {
  // The tag, not a value: eight raw bytes both sides handle before the
  // described walk, and so deliberately outside it.
  std::string_view magic;
  std::uint32_t format = 0;
  std::uint32_t schema = 0;
  std::uint8_t scalar_size = 0;
  std::uint8_t scalar_kind = 0; // 1 floating point, 0 otherwise
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

struct Container;

// The file and its prologue:
class Store {
  friend struct Container;

  static DDX_API void put_header(const FileHeader &h, std::span<std::byte> into);
  static DDX_API result<FileHeader> get_header(std::span<const std::byte> bytes,
                                               std::string_view magic);
  static DDX_API std::uint32_t checksum(std::span<const std::byte> bytes);
  static DDX_API result<std::vector<std::byte>>
  read_file(const std::filesystem::path &path);
  // Staged beside the target and renamed over it: a reader never sees a
  // half-written file, and a failed save leaves the old one standing.
  static DDX_API result<void> write_file(const std::filesystem::path &path,
                                         std::span<const std::byte> bytes);
};

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
  [[nodiscard]] static DDX_API result<void>
  write(const std::filesystem::path &path, std::span<const std::byte> bytes);

  // A payload with a described shape, encoded exactly as every other one is.
  template <typename V>
  [[nodiscard]] static std::vector<std::byte> encode(const V &v) {
    std::vector<std::byte> out;
    Writer w{out};
    wire(w, std::as_const(v));
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
};

} // namespace ddx::rt::detail
