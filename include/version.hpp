#ifndef DIFF_VERSION_HPP
#define DIFF_VERSION_HPP

#define DIFF_VERSION_MAJOR 1
#define DIFF_VERSION_MINOR 1
#define DIFF_VERSION_PATCH 0

#define DIFF_VERSION                                                           \
  (DIFF_VERSION_MAJOR * 100000 + DIFF_VERSION_MINOR * 100 + DIFF_VERSION_PATCH)

#define DIFF_STRINGIZE_(x) #x
#define DIFF_STRINGIZE(x) DIFF_STRINGIZE_(x)

#define DIFF_LIB_VERSION                                                       \
  DIFF_STRINGIZE(DIFF_VERSION_MAJOR) "_" DIFF_STRINGIZE(DIFF_VERSION_MINOR)

#endif // DIFF_VERSION_HPP
