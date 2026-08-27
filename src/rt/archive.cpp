// The part of the archive carrying no scalar: prologue, checksum, file and
// opcode label table.  Out of line like coupling.cpp and dot.cpp, or it would be
// emitted once per scalar for byte-identical code.
#include "rt/archive.hpp"

#include "util/ranges.hpp"

#include <boost/crc.hpp>
#include <boost/endian/conversion.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <random>
#include <ranges>
#include <system_error>
#include <thread>

namespace ddx::rt {
namespace {

// Fixed offsets, not a struct laid over the bytes: the on-disk prologue is not
// this machine's padding.
inline constexpr std::size_t off_magic = 0;
inline constexpr std::size_t off_format = 8;
inline constexpr std::size_t off_schema = 12;
inline constexpr std::size_t off_opcodes = 16;
inline constexpr std::size_t off_scalar_size = 18;
inline constexpr std::size_t off_scalar_kind = 19;
inline constexpr std::size_t off_model_nodes = 20;
inline constexpr std::size_t off_model_digest = 24;
inline constexpr std::size_t off_payload_bytes = 32;
inline constexpr std::size_t off_payload_crc = 40;
inline constexpr std::size_t off_reserved = 44; // 12 bytes, must be zero

template <typename U>
void put(std::span<std::byte> into, std::size_t at, U v) {
  const auto w = boost::endian::native_to_little(v);
  std::memcpy(into.data() + at, &w, sizeof w);
}

// Per-process and per-call, in standard C++ rather than getpid(): drawn once so
// it is not paid per write, counted so one process cannot collide with itself.
[[nodiscard]] std::string stage_suffix() {
  static const auto salt = std::random_device{}();
  static std::atomic<std::uint32_t> seq{0};
  return std::format(".{:08x}{:08x}.tmp", salt, seq.fetch_add(1));
}

// Windows will not replace a destination that another handle still has open,
// and racing writers are exactly that: MoveFileEx opens the target for delete,
// so the loser gets a sharing violation instead of losing an atomic
// last-writer-wins.  Every hold is brief -- another rename in flight, or a
// scanner on a freshly written file -- so a bounded retry restores the property
// the staging name was chosen for.  POSIX rename never fails this way, and
// there the loop never comes round.
//
// Retried whatever the error says, rather than on the sharing violation alone:
// the staged file is written and closed and sits in the target's own directory,
// so the remaining ways to fail here are exotic, and one that cannot recover
// only spends the budget before failing exactly as it would have.
[[nodiscard]] bool rename_over(const std::filesystem::path &from,
                               const std::filesystem::path &to) {
  using namespace std::chrono_literals;
  auto backoff = 1ms;
  for (int attempt = 0; attempt < 20; ++attempt) {
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (!ec) {
      return true;
    }
    // Yielding first: the common contender is a rename mid-flight, and
    // sleeping a whole tick for it would cost more than the write did.
    if (attempt < 4) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(backoff);
      backoff = std::min(backoff * 2, 32ms);
    }
  }
  return false;
}

template <typename U>
[[nodiscard]] U get(std::span<const std::byte> from, std::size_t at) {
  U w{};
  std::memcpy(&w, from.data() + at, sizeof w);
  return boost::endian::little_to_native(w);
}

} // namespace

void put_header(const FileHeader &h, std::span<std::byte> into) {
  assert(into.size() >= header_bytes);
  std::ranges::fill(into.first(header_bytes), std::byte{});
  std::memcpy(into.data() + off_magic, h.magic.data(),
              std::min(h.magic.size(), std::size_t{8}));
  put(into, off_format, h.format);
  put(into, off_schema, h.schema);
  put(into, off_opcodes, h.opcodes);
  put(into, off_scalar_size, h.scalar_size);
  put(into, off_scalar_kind, h.scalar_kind);
  put(into, off_model_nodes, h.model_nodes);
  put(into, off_model_digest, h.model_digest);
  put(into, off_payload_bytes, h.payload_bytes);
  put(into, off_payload_crc, h.payload_crc);
}

result<FileHeader> get_header(std::span<const std::byte> bytes,
                              std::string_view magic) {
  if (bytes.size() < header_bytes) {
    return fail(errc::bad_archive);
  }
  if (std::memcmp(bytes.data() + off_magic, magic.data(),
                  std::min(magic.size(), std::size_t{8})) != 0) {
    return fail(errc::bad_archive);
  }
  // Letting a nonzero reserved byte past would make claiming it later a silent
  // format change.
  if (!std::ranges::all_of(bytes.subspan(off_reserved, header_bytes -
                                                           off_reserved),
                           [](std::byte b) { return b == std::byte{}; })) {
    return fail(errc::bad_archive);
  }

  const FileHeader h{.magic = magic,
                     .format = get<std::uint32_t>(bytes, off_format),
                     .schema = get<std::uint32_t>(bytes, off_schema),
                     .opcodes = get<std::uint16_t>(bytes, off_opcodes),
                     .scalar_size = get<std::uint8_t>(bytes, off_scalar_size),
                     .scalar_kind = get<std::uint8_t>(bytes, off_scalar_kind),
                     .model_nodes = get<std::uint32_t>(bytes, off_model_nodes),
                     .model_digest =
                         get<std::uint64_t>(bytes, off_model_digest),
                     .payload_bytes =
                         get<std::uint64_t>(bytes, off_payload_bytes),
                     .payload_crc = get<std::uint32_t>(bytes, off_payload_crc)};

  // The payload cannot be longer than the file that carries it, checked before
  // anything sizes a buffer from it.  Depends on the `bytes.size() <
  // header_bytes` rejection above: the subtraction is unsigned, so on a shorter
  // file it underflows and this passes for any length at all.
  if (h.payload_bytes > bytes.size() - header_bytes) {
    return fail(errc::archive_corrupt);
  }
  return h;
}

std::uint32_t checksum(std::span<const std::byte> bytes) {
  boost::crc_32_type crc;
  crc.process_bytes(bytes.data(), bytes.size());
  return crc.checksum();
}

result<std::vector<std::byte>> read_file(const std::filesystem::path &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || !std::filesystem::is_regular_file(path, ec)) {
    return fail(errc::archive_io);
  }

  if (std::ifstream in{path, std::ios::binary}; !in) {
    return fail(errc::archive_io);
  } else {
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
      return fail(errc::archive_io);
    }
    return bytes;
  }

}

result<void> write_file(const std::filesystem::path &path,
                        std::span<const std::byte> bytes) {
  // Beside the target: rename is only atomic within a filesystem, and one
  // across a mount boundary would silently become a copy.  The suffix is unique
  // per writer -- with a shared ".tmp" two processes truncate each other's
  // partial write and both rename it.  Distinct names make the loser's rename a
  // no-op: last writer wins, and every reader sees one whole file.
  std::filesystem::path tmp = path;
  tmp += stage_suffix();
  std::error_code ec;
  {
    std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
    if (!out) {
      return fail(errc::archive_io);
    }
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
      std::filesystem::remove(tmp, ec);
      return fail(errc::archive_io);
    }
  }
  if (!rename_over(tmp, path)) {
    std::filesystem::remove(tmp, ec);
    return fail(errc::archive_io);
  }
  return {};
}

std::vector<std::string> opcode_labels() {
  return detail::op_info |
         std::views::transform(
             [](const detail::OpInfo &i) { return std::string{i.label}; }) |
         impl::to<std::vector<std::string>>();
}

std::optional<OpCode> opcode_of(std::string_view label) {
  // opcode_of_label() is the same lookup, consteval; a loader needs it at run
  // time.
  const auto row =
      std::ranges::find(detail::op_info, label, &detail::OpInfo::label);
  return row == detail::op_info.end()
             ? std::nullopt
             : std::optional{static_cast<OpCode>(row - detail::op_info.begin())};
}

} // namespace ddx::rt
