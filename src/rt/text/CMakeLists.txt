# The grammar, and the only object in libddx compiled with exceptions.
# Boost.Parser wraps every parse() in a try/catch and has no BOOST_NO_EXCEPTIONS
# path, so it cannot compile under the flag the rest of this tree uses; parse()
# catches everything before it returns, which is what keeps the difference to
# this one file.  An OBJECT library rather than a source on ddx_rt, because only
# a target of its own can carry its own flags.
add_library(ddx_text_objs OBJECT parse.cpp)

set_target_properties(ddx_text_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

# BOOST_PARSER_DISABLE_TRACE: Boost.Parser emits trace code for every parser it
# instantiates, and this grammar has no debugging story that wants it.
target_compile_definitions(ddx_text_objs PRIVATE DDX_BUILDING BOOST_PARSER_DISABLE_TRACE)
# ddx_rt's header dependencies, not ddx_rt: these link *into* it.
target_link_libraries(ddx_text_objs PRIVATE ddx Boost::headers)

# Boost.Parser's parse() is a try/catch, so this one object keeps exceptions.
ddx_target_flags(ddx_text_objs EXCEPTIONS)

target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_text_objs>)
