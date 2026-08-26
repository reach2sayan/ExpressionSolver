#include "rt/archive.hpp"
#include "rt/equation.hpp"
#include "rt/expressions.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ios>
#include <filesystem>
#include <latch>
#include <random>
#include <thread>
#include <ranges>
#include <span>
#include <string>
#include <system_error>
#include <vector>

// Saving and loading a built equation.  The file carries the arena and the
// sweeps; what it must reproduce is every answer, to the bit -- these compare a
// loaded equation against a freshly built one in the same binary, which is the
// only comparison the key is good for (node ids are assigned in construction
// order, so another compiler numbers the same model differently).
namespace {

// A file that removes itself, since a failing EXPECT must not leave one behind
// for the next run to load.
//
// The name carries a per-process token.  Without one, two of these binaries run
// at once -- a sharded suite, a `ctest --repeat`, two checkouts sharing /tmp --
// write and delete each other's files, and every save fails for reasons that
// have nothing to do with what is being tested.  A fixed path in the system
// temp directory is shared mutable state between processes, however private it
// looks.
class Scratch {
public:
  explicit Scratch(std::string name)
      : path_(std::filesystem::temp_directory_path() /
              ("ddx_archive_" + std::move(name) + "_" + token() + ".ddx")) {
    wipe();
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  ~Scratch() { wipe(); }

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }
  operator const std::filesystem::path &() const { return path_; } // NOLINT

  [[nodiscard]] std::vector<std::byte> bytes() const {
    // Checked, not dereferenced: `*` on a failed result is undefined, so a
    // fixture bug that made the file vanish would surface as a segfault in the
    // test rather than as the failure it is.  That is how the shared-path
    // problem above presented.
    auto read = ddx::rt::read_file(path_);
    EXPECT_TRUE(read.has_value()) << "scratch file went missing";
    return read ? std::move(*read) : std::vector<std::byte>{};
  }
  void write(std::span<const std::byte> b) const {
    ASSERT_TRUE(ddx::rt::write_file(path_, b).has_value());
  }

private:
  [[nodiscard]] static const std::string &token() {
    static const std::string t = std::to_string(std::random_device{}());
    return t;
  }
  void wipe() const {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  std::filesystem::path path_;
};

// Coupled, so the Hessian colouring is real work rather than a formality --
// which is the expensive thing the file exists to keep.
auto coupled() {
  return ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return exp(x * y) + log(y * z) * sin(x + z) + z * z * x;
  });
}

// The same model, written the same way.  Not `auto f = coupled()` twice by
// accident: these must be two independent builds over two arenas.
constexpr std::array kPoint{1.3, 0.7, 2.1};

void same_answers(const auto &want, const auto &got) {
  ASSERT_EQ(want.arity(), got.arity());
  ASSERT_EQ(want.symbols()->size(), got.symbols()->size());
  EXPECT_TRUE(std::ranges::equal(*want.symbols(), *got.symbols()));

  // Bit-identical, not near: a loaded graph is the same graph, so the same
  // operations happen in the same order and nothing may round differently.
  EXPECT_EQ(*want.evaluate(kPoint), *got.evaluate(kPoint));
  EXPECT_EQ(*want.jacobian(kPoint), *got.jacobian(kPoint));
  EXPECT_EQ(*want.hessian(kPoint), *got.hessian(kPoint));
}

} // namespace

TEST(RtArchive, RoundTripsEveryDerivative) {
  const Scratch file{"roundtrip"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  same_answers(built, *loaded);
  EXPECT_TRUE(loaded->loaded());
  EXPECT_FALSE(built.loaded());
}

TEST(RtArchive, RoundTripsASystem) {
  const Scratch file{"system"};
  const auto built = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::array{x * y + sin(x), exp(x) - y * y};
  });
  ASSERT_TRUE(built.save(file).has_value());

  const auto loaded = ddx::rt::load<double, 2>(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  EXPECT_EQ(*built.evaluate(1.3, 0.7), *loaded->evaluate(1.3, 0.7));
  EXPECT_EQ(*built.jacobian(1.3, 0.7), *loaded->jacobian(1.3, 0.7));
}

// The output count is in the type, so a file holding some other number of
// functions has to be refused rather than reinterpreted -- reading a
// two-function file as one would size every caller's buffer wrongly.
TEST(RtArchive, RefusesTheWrongOutputCount) {
  const Scratch file{"outputs"};
  const auto two = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return std::array{x * x, sin(x)};
  });
  ASSERT_TRUE(two.save(file).has_value());

  const auto as_one = ddx::rt::load<double, 1>(file.path());
  ASSERT_FALSE(as_one.has_value());
  EXPECT_EQ(as_one.error().code, ddx::errc::archive_mismatch);
}

TEST(RtArchive, VerifyAnswersBothQuestions) {
  const Scratch file{"verify"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  // Readable by this build...
  EXPECT_TRUE(ddx::rt::verify(file.path()).has_value());
  // ...and it is this equation.
  EXPECT_TRUE(built.verify(file.path()).has_value());

  // A different model over the same symbols: the file still loads, and still
  // is not this one.
  const auto other = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return x + y + z;
  });
  EXPECT_TRUE(ddx::rt::verify(file.path()).has_value());
  const auto mismatch = other.verify(file.path());
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_EQ(mismatch.error().code, ddx::errc::archive_mismatch);
}

TEST(RtArchive, MissingFileIsNotCorruption) {
  const Scratch file{"missing"}; // constructed, deliberately never written
  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  // The first run of a cache, not a damaged file: a caller building one acts
  // on the difference.
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_io);
}

// Nothing below may abort.  Under -fno-exceptions a Boost throw would, which
// is exactly why the checksum and the invariants are cleared before a byte is
// trusted -- these are the cases that prove the order holds.
TEST(RtArchive, RefusesATruncatedFile) {
  const Scratch file{"truncated"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  // Just past the prologue, so the header parses and the payload does not.
  auto cut = whole;
  cut.resize(ddx::rt::header_bytes + 8);
  file.write(cut);
  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_TRUE(loaded.error().code == ddx::errc::archive_corrupt ||
              loaded.error().code == ddx::errc::bad_archive);
}

TEST(RtArchive, RefusesAFlippedByte) {
  const Scratch file{"flipped"};
  ASSERT_TRUE(coupled().save(file).has_value());
  auto bytes = file.bytes();

  // One bit, in the middle of the payload, where it will land inside a node.
  const std::size_t at = ddx::rt::header_bytes +
                         (bytes.size() - ddx::rt::header_bytes) / 2;
  bytes[at] ^= std::byte{0x01};
  file.write(bytes);

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

TEST(RtArchive, RefusesForeignAndFutureFiles) {
  const Scratch file{"foreign"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  auto wrong_magic = whole;
  wrong_magic[0] = std::byte{'x'};
  file.write(wrong_magic);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);

  // A format counter this build does not read.  Bumped in place: the checksum
  // covers the payload, not the prologue, so this is what a future writer's
  // file looks like to us.
  auto future = whole;
  future[8] = std::byte{0xFF};
  file.write(future);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);

  // A reserved byte claimed by nobody yet.  Letting one through would make
  // claiming it later a silent format change.
  auto reserved = whole;
  reserved[44] = std::byte{0x01};
  file.write(reserved);
  EXPECT_EQ(ddx::rt::load(file.path()).error().code, ddx::errc::bad_archive);
}

TEST(RtArchive, RefusesTheWrongScalar) {
  const Scratch file{"scalar"};
  ASSERT_TRUE(coupled().save(file).has_value());
  // Same file, read as float: the scalar tag and the schema stamp both say no.
  EXPECT_EQ(ddx::rt::verify<float>(file.path()).error().code,
            ddx::errc::bad_archive);
}

// Concurrent writers to one path all succeed, and the file is one of theirs.
//
// What this actually catches, measured rather than assumed: with a shared
// staging name the *losing rename fails*.  A writer renames `path.tmp` over the
// target, and the next one finds nothing left to rename -- so a legitimate
// write reports an error.  Reverting stage_suffix() to a bare ".tmp" produces
// ~900 failed writes out of 1280 here, every time.
//
// It is NOT a torn-file test, though the assertions below would catch one.
// Tearing was not reachable at all: distinct payloads at 1 KiB, 64 KiB and
// 1 MiB, forty rounds each, gave 0 torn files and 40/40 whole ones even with
// the bug present, because rename is atomic and the content race resolves
// before it.  The whole-file and checksum checks are kept -- they cost nothing
// and state the property the function is *for* -- but the failing-write
// assertion is what does the work, and a future reader should not conclude
// from a green run that tearing is covered.  It is not, and on this platform it
// may not be reachable to cover.
//
// Threads rather than processes because they share the staging name the same
// way and a test can actually join them.
TEST(RtArchive, ConcurrentWritersToOnePathAllSucceed) {
  const Scratch file{"racing"};
  constexpr std::size_t kWriters = 8;
  static_assert(kWriters >= 2, "one writer races nobody");

  // Built up front: build_hessian_impl appends to its arena, so an equation is
  // not something two threads may construct over one Builder.  Each thread gets
  // a distinct model, so the file it wrote is identifiable afterwards.
  std::vector<std::vector<std::byte>> payloads;
  for (std::size_t k = 0; k < kWriters; ++k) {
    const auto eq = ddx::rt::equation([k] {
      const auto x = ddx::rt::var("x");
      return x * static_cast<double>(k + 1);
    });
    const Scratch one{"racing_src_" + std::to_string(k)};
    EXPECT_TRUE(eq.save(one).has_value());
    payloads.push_back(one.bytes());
  }

  // Everything below rests on the payloads being pairwise distinct: that is
  // what makes a torn file detectable at all, since a mixture of two identical
  // ones is identical to both.  Assert it rather than trust that eight models
  // written to look different still are -- `x * 1.0` folds to `x`, and a later
  // tidy-up of these models could quietly make two of them agree and leave
  // every other assertion here green forever.
  //
  // The overlap is *forced*, below, rather than asserted.  Asserting it would
  // be scheduling-dependent and go intermittent on a loaded runner; leaving it
  // to chance would let the writers serialise one day and the test would pass
  // having exercised nothing concurrent.  A latch makes "they all write at
  // once" a fact about the code instead of about the scheduler, and needs no
  // assertion at all.
  for (std::size_t a = 0; a < payloads.size(); ++a) {
    for (std::size_t b = a + 1; b < payloads.size(); ++b) {
      ASSERT_NE(payloads[a], payloads[b]) << "writers " << a << " and " << b;
    }
  }

  std::latch start{static_cast<std::ptrdiff_t>(kWriters)};
  std::vector<std::jthread> writers;
  writers.reserve(kWriters);
  for (const auto &bytes : payloads) {
    writers.emplace_back([&file, &bytes, &start] {
      start.arrive_and_wait(); // nobody writes until everybody is ready
      for (int spin = 0; spin < 8; ++spin) {
        EXPECT_TRUE(ddx::rt::write_file(file.path(), bytes).has_value());
      }
    });
  }
  writers.clear(); // join

  // The stronger property, which this test states but does not reliably
  // exercise -- see the note above.
  const auto landed = file.bytes();
  EXPECT_TRUE(std::ranges::any_of(
      payloads, [&landed](const auto &p) { return p == landed; }))
      << "the file is not any single writer's";
  const auto loaded = ddx::rt::load(file.path());
  EXPECT_TRUE(loaded.has_value())
      << "a torn file would fail the checksum: " << loaded.error().code;

  // And nothing was left staged beside it.
  std::error_code ec;
  std::size_t leftovers = 0;
  for (const auto &entry :
       std::filesystem::directory_iterator{file.path().parent_path(), ec}) {
    const auto name = entry.path().filename().string();
    if (name.starts_with(file.path().filename().string()) &&
        name.ends_with(".tmp")) {
      ++leftovers;
    }
  }
  EXPECT_EQ(leftovers, 0u) << "staging files left behind";
}

// --- the cache ---------------------------------------------------------------

TEST(RtArchive, CacheBuildsThenLoads) {
  const Scratch file{"cache"};
  const auto model = [] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x * y) + sin(x) * log(y);
  };

  const auto first = ddx::rt::equation(file.path(), model);
  EXPECT_FALSE(first.loaded()) << "nothing to load on the first run";
  ASSERT_TRUE(std::filesystem::exists(file.path())) << "the first run saves";

  const auto second = ddx::rt::equation(file.path(), model);
  EXPECT_TRUE(second.loaded()) << "the second run takes the sweeps off disk";
  EXPECT_EQ(*first.evaluate(1.3, 0.7), *second.evaluate(1.3, 0.7));
  EXPECT_EQ(*first.jacobian(1.3, 0.7), *second.jacobian(1.3, 0.7));
  EXPECT_EQ(*first.hessian(1.3, 0.7), *second.hessian(1.3, 0.7));
}

TEST(RtArchive, CacheRebuildsWhenTheModelChanges) {
  const Scratch file{"stale"};
  const auto before = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x;
  });
  EXPECT_FALSE(before.loaded());

  // The same symbols, a different model: the key is the arena, so this must
  // rebuild and overwrite rather than load what is there.
  const auto after = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x * x;
  });
  EXPECT_FALSE(after.loaded()) << "an edited model is not a cache hit";
  EXPECT_NEAR(*after.evaluate(2.0), 8.0, 1e-15);

  // And what it wrote is now the current model.
  const auto again = ddx::rt::equation(file.path(), [] {
    const auto x = ddx::rt::var("x");
    return x * x * x;
  });
  EXPECT_TRUE(again.loaded()) << "the rebuild overwrote the stale file";
  EXPECT_NEAR(*again.evaluate(2.0), 8.0, 1e-15);
}

TEST(RtArchive, CacheSurvivesACorruptFile) {
  const Scratch file{"cache_corrupt"};
  const auto model = [] {
    const auto x = ddx::rt::var("x");
    return sin(x) * x;
  };
  ASSERT_FALSE(ddx::rt::equation(file.path(), model).loaded());

  auto bytes = file.bytes();
  bytes[bytes.size() - 3] ^= std::byte{0xFF};
  file.write(bytes);

  // A damaged cache is a rebuild, never a refusal and never an abort.
  const auto rebuilt = ddx::rt::equation(file.path(), model);
  EXPECT_FALSE(rebuilt.loaded());
  EXPECT_NEAR(*rebuilt.evaluate(1.3), std::sin(1.3) * 1.3, 1e-15);
  EXPECT_TRUE(ddx::rt::equation(file.path(), model).loaded())
      << "and it left a good file behind";
}

// Every byte of a real file, corrupted every way a single byte can be.  A
// parser that reads a length out of the payload is exactly the kind that walks
// off the end when the length is wrong, and there is no throw here to catch it
// -- a Boost one would abort.  So the claim is not that these are refused with
// the right code, only that every one of them *returns*.
//
// Deterministic: a fixed generator over fixed offsets, so a failure is a
// failure every run rather than once a week on someone else's machine.
TEST(RtArchive, SurvivesEverySingleByteCorruption) {
  const Scratch file{"fuzz"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();
  ASSERT_GT(whole.size(), ddx::rt::header_bytes);

  // Every offset would be a few hundred thousand loads; a stride samples the
  // prologue densely and the payload throughout.
  std::size_t refused = 0;
  std::size_t accepted = 0;
  for (std::size_t at = 0; at < whole.size();
       at += (at < ddx::rt::header_bytes ? 1 : 97)) {
    for (const std::byte mask : {std::byte{0x01}, std::byte{0x80},
                                 std::byte{0xFF}}) {
      auto bytes = whole;
      bytes[at] ^= mask;
      file.write(bytes);
      // The only requirement: it comes back.  Under ASan this is also where a
      // read past the payload would be caught.
      if (const auto loaded = ddx::rt::load(file.path())) {
        ++accepted;
      } else {
        ++refused;
      }
    }
  }
  // What this sweep does and does not prove.  It proves every single-byte
  // corruption *returns* rather than walking off the end.  It does not prove
  // the file's meaning is protected: measured, 390 of 4290 corruptions get past
  // the checksum, and none of them happened to be the one bit in the one label
  // that silently remapped an opcode -- see RefusesAFlippedOpcodeLabel, which
  // had to be reasoned out rather than stumbled on.
  EXPECT_GT(refused, 0u);
  EXPECT_EQ(refused + accepted, 3 * (ddx::rt::header_bytes +
                                     (whole.size() - ddx::rt::header_bytes +
                                      96) / 97));
}

// Every prefix of a real file, which is what a partial write leaves behind.
TEST(RtArchive, SurvivesEveryTruncation) {
  const Scratch file{"truncations"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();

  for (std::size_t keep = 0; keep < whole.size(); keep += 13) {
    auto cut = whole;
    cut.resize(keep);
    file.write(cut);
    // No prefix of a valid file is a valid file: the payload length in the
    // prologue no longer matches what follows it.
    EXPECT_FALSE(ddx::rt::load(file.path()).has_value()) << "kept " << keep;
  }
}

// A payload whose checksum agrees, and whose contents are still a lie.
//
// This is the case neither fuzz loop above can reach, and the distinction is
// the point: they *damage* a file, and damage is caught by the checksum long
// before anything parses it.  Everything past the checksum is untested by
// construction unless a test forges a consistent payload instead -- so this one
// writes through the real serialiser, which recomputes the CRC.
//
// A colouring's `count` is read straight from the file and is what indexes the
// two colours-by-n tables.  `count * nsym` is where a forged count wraps and
// agrees with a length it has no business agreeing with, and the graph then
// carries a colour count of 4.6e18 into every consumer that walks one.  Only an
// even symbol count can do it: with an odd nsym the multiply is invertible
// modulo 2^64, so no second count exists.
TEST(RtArchive, RefusesAForgedColouringCount) {
  const Scratch file{"forged"};
  const auto four = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto c = ddx::rt::var("c");
    const auto d = ddx::rt::var("d");
    return exp(a * b) + log(c * d) * sin(a + c) + b * d;
  });
  ASSERT_TRUE(four.save(file).has_value());

  auto snap = ddx::rt::load_snapshot(file.path());
  ASSERT_TRUE(snap.has_value());
  auto &coloring = snap->hessians.front().coloring;
  const std::size_t nsym = snap->symbols.size();
  ASSERT_EQ(nsym % 2, 0u) << "an odd symbol count cannot wrap";

  const std::size_t forged = coloring.count + (std::size_t{1} << 62);
  ASSERT_EQ(forged * nsym, coloring.scatter.size()) << "the wrap must agree";
  ASSERT_NE(forged, coloring.count);
  coloring.count = forged;

  // Through the real serialiser, so the checksum agrees with the lie.
  ASSERT_TRUE(ddx::rt::save(*snap, file.path()).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_FALSE(loaded.has_value()) << "a wrapped count agreed with a length";
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

// The opcode table is what every opcode byte in the payload *means*, so it is
// inside the checksum.  It was not, at format 1, and this is the case that
// found it: '+' is 0x2B and '/' is 0x2F -- one bit apart, and both binary, so
// remapping every Add to a Div passes sound()'s arity check untouched.
//
// The model deliberately contains no '+' of its own.  Every Add in it is one
// the reverse sweep created, which puts it *above* model_nodes and so beyond
// the reach of the header's model digest -- the defence that catches this same
// flip when it lands in the model itself.  Both halves are needed: the digest
// covers the model, the checksum covers everything.
//
// Not reachable by the byte-flip sweep above, which is why that sweep did not
// find it: this needs a specific bit in a specific label.
TEST(RtArchive, RefusesAFlippedOpcodeLabel) {
  const Scratch file{"label"};
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return sin(x) * cos(x);
  });
  ASSERT_TRUE(eq.save(file).has_value());
  const auto want = *eq.jacobian(0.7);

  auto bytes = file.bytes();
  bool flipped = false;
  for (std::size_t i = ddx::rt::header_bytes; i + 9 < bytes.size(); ++i) {
    std::uint64_t len = 0;
    std::memcpy(&len, bytes.data() + i, sizeof len);
    if (len != 1 || static_cast<char>(bytes[i + 8]) != '+') {
      continue;
    }
    bytes[i + 8] ^= std::byte{0x04}; // '+' -> '/'
    flipped = true;
    break;
  }
  ASSERT_TRUE(flipped) << "no one-character \"+\" label in the table";
  file.write(bytes);

  const auto loaded = ddx::rt::load(file.path());
  if (loaded) {
    // Precise, so a future reader sees what was at stake rather than only that
    // something was refused.
    EXPECT_EQ(*loaded->jacobian(0.7), want)
        << "an opcode was remapped and the gradient moved";
    FAIL() << "a flipped opcode label was accepted";
  }
  EXPECT_EQ(loaded.error().code, ddx::errc::archive_corrupt);
}

// The prologue is the one region a checksum cannot cover, because the checksum
// lives in it.  So every field it carries is verified field by field, and no
// field is written that is not verified -- inert-but-unprotected is the state a
// field is in immediately before it is read, trusted, and wrong.
//
// Swept rather than spot-checked, deliberately: this is what catches the *next*
// prologue field somebody adds and forgets to check.  It found `model_nodes`,
// which was duplicated in the payload under the checksum and so read back by
// nobody -- 12 of 168 corruptions accepted, all four of its bytes.
//
// Exhaustive rather than a few masks, also deliberately.  A sampled mask set is
// only adequate while every field is checked by *exact* equality against a
// value derived elsewhere, which is true today -- and a field checked against a
// range instead, which is the natural shape of a new one, can accept a value
// that some masks never produce and others do.  Rather than write that
// precondition down and hope, the sweep tries all 255 corruptions of every
// byte, so its adequacy does not depend on how the next field gets checked.
// 1.5 s, nearly all of it writing files.
TEST(RtArchive, EveryPrologueByteIsVerified) {
  const Scratch file{"prologue"};
  ASSERT_TRUE(coupled().save(file).has_value());
  const auto whole = file.bytes();
  ASSERT_GT(whole.size(), ddx::rt::header_bytes);

  for (std::size_t at = 0; at < ddx::rt::header_bytes; ++at) {
    for (int mask = 1; mask < 256; ++mask) {
      auto bytes = whole;
      bytes[at] ^= static_cast<std::byte>(mask);
      file.write(bytes);
      ASSERT_FALSE(ddx::rt::load(file.path()).has_value())
          << "prologue byte " << at << " accepts corruption 0x" << std::hex
          << mask << ": written but not fully verified";
    }
  }
}

// --- the pieces --------------------------------------------------------------

// Opcode bytes are table-order, so the file names them by label and remaps on
// load.  That is what keeps a file readable when a transcendental is appended
// to DDX_UNARY_MATH_TABLE and every enumerator above it shifts.
TEST(RtArchive, OpcodesTravelByLabel) {
  const auto labels = ddx::rt::opcode_labels();
  ASSERT_EQ(labels.size(), ddx::rt::op_count);
  for (const auto [i, label] : labels | std::views::enumerate) {
    const auto back = ddx::rt::opcode_of(label);
    ASSERT_TRUE(back.has_value()) << label;
    EXPECT_EQ(static_cast<std::size_t>(*back), static_cast<std::size_t>(i));
  }
  EXPECT_FALSE(ddx::rt::opcode_of("no such op").has_value());
}

// The stamp is what makes a changed struct refuse old files rather than
// misread them, so it must actually depend on the fields.
TEST(RtArchive, TheSchemaStampSeparatesShapes) {
  constexpr auto snapshot = ddx::rt::detail::schema_of<ddx::rt::Snapshot<double>>();
  constexpr auto narrower = ddx::rt::detail::schema_of<ddx::rt::Snapshot<float>>();
  constexpr auto unrelated = ddx::rt::detail::schema_of<ddx::rt::Coloring>();
  static_assert(snapshot != narrower);
  static_assert(snapshot != unrelated);
  EXPECT_NE(snapshot, 0u);
}

// A saved arena is installed verbatim rather than replayed: make() folds and
// swaps a commutative pair, so a replay would renumber the very ids the saved
// Jacobian and colouring name.
TEST(RtArchive, LoadedIdsAreTheSavedIds) {
  const Scratch file{"ids"};
  const auto built = coupled();
  ASSERT_TRUE(built.save(file).has_value());

  auto snap = ddx::rt::load_snapshot(file.path());
  ASSERT_TRUE(snap.has_value()) << snap.error().code;

  // rebuild() consumes the node array -- it becomes the arena's -- so the
  // comparison keeps its own copy.
  const auto saved = snap->nodes;
  const auto arena = ddx::rt::rebuild(*snap);
  ASSERT_EQ(arena->size(), saved.size());
  for (const auto [i, node] : saved | std::views::enumerate) {
    const auto &back = (*arena)[static_cast<ddx::rt::NodeId>(i)];
    EXPECT_EQ(back.op, node.op) << "node " << i;
    EXPECT_EQ(back.a, node.a) << "node " << i;
    EXPECT_EQ(back.b, node.b) << "node " << i;
    EXPECT_EQ(back.slot, node.slot) << "node " << i;
  }
}

// --- the machine code --------------------------------------------------------
//
// The graph half of a file saves milliseconds; this half saves a compile, which
// at these sizes is hundreds of them.  A stored kernel is only ever run when
// the graph, the host and the options it was emitted under all still agree --
// adopt() cannot tell that an object came from another graph, so nothing
// reaches it that has not been checked first.
#ifdef DDX_HAS_JIT
TEST(RtArchive, CarriesTheCompiledKernel) {
  const Scratch file{"kernel"};
  {
    auto built = coupled();
    // Off by default: a kernel does not keep a megabyte of machine code alive
    // on the chance that someone saves it.
    built.options({.backend = ddx::jit::Backend::Compile,
                   .retain_object = true});
    ASSERT_TRUE(built.wait_for_kernel());
    ASSERT_TRUE(built.save(file).has_value());
  }

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  // No wait_for_kernel(): the kernel was linked from the file during the
  // freeze, so it is already answering.  That is the whole claim.
  EXPECT_TRUE(loaded->uses_kernel());
  EXPECT_TRUE(loaded->kernel_level().has_value());

  // And it is the same kernel, not merely an equivalent one.
  const auto want = coupled();
  EXPECT_EQ(*loaded->jacobian(kPoint), *want.jacobian(kPoint));
  EXPECT_EQ(*loaded->hessian(kPoint), *want.hessian(kPoint));
}

TEST(RtArchive, SavesNoKernelWithoutRetainObject) {
  const Scratch file{"no_retain"};
  auto built = coupled();
  built.options({.backend = ddx::jit::Backend::Compile});
  ASSERT_TRUE(built.wait_for_kernel());
  ASSERT_TRUE(built.save(file).has_value());

  // The graph is there and the code is not, which is a shortfall in what the
  // file is worth rather than a failure.
  const auto snap = ddx::rt::load_snapshot(file.path());
  ASSERT_TRUE(snap.has_value());
  EXPECT_TRUE(snap->objects.empty());
  EXPECT_TRUE(ddx::rt::load(file.path()).has_value());
}

TEST(RtArchive, RefusesAKernelFromAnotherGraph) {
  const Scratch file{"foreign_kernel"};
  auto built = coupled();
  built.options({.backend = ddx::jit::Backend::Compile,
                 .retain_object = true});
  ASSERT_TRUE(built.wait_for_kernel());

  ASSERT_TRUE(built.save(file).has_value());

  // The right code, relabelled as some other graph's, and written back through
  // the same serialiser so the checksum still holds.  A stored kernel run
  // against a graph it was not emitted from is silently wrong arithmetic, so
  // the digest is what has to gate it -- not the checksum, which would happily
  // agree.
  auto snap = ddx::rt::load_snapshot(file.path());
  ASSERT_TRUE(snap.has_value());
  ASSERT_FALSE(snap->objects.empty());
  for (auto &o : snap->objects) {
    o.digest ^= 1;
  }
  ASSERT_TRUE(ddx::rt::save(*snap, file.path()).has_value());

  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value());
  // The graph still loads; the code is simply not adopted.
  EXPECT_EQ(*loaded->jacobian(kPoint), *coupled().jacobian(kPoint));
}
#endif
