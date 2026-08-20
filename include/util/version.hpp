#ifndef DDX_VERSION_HPP
#define DDX_VERSION_HPP

#define DDX_VERSION_MAJOR 1
#define DDX_VERSION_MINOR 1
#define DDX_VERSION_PATCH 0

#define DDX_VERSION                                                           \
  (DDX_VERSION_MAJOR * 100000 + DDX_VERSION_MINOR * 100 + DDX_VERSION_PATCH)

#define DDX_STRINGIZE_(x) #x
#define DDX_STRINGIZE(x) DDX_STRINGIZE_(x)

#define DDX_LIB_VERSION                                                       \
  DDX_STRINGIZE(DDX_VERSION_MAJOR) "_" DDX_STRINGIZE(DDX_VERSION_MINOR)

#endif // DDX_VERSION_HPP
