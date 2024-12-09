#ifndef VECTORS_H
#define VECTORS_H

#include "Helper.h"

typedef uint8_t uchar_t;
typedef int8_t char_t;

typedef uint16_t ushort_t;
typedef int16_t short_t;

typedef uint32_t uint_t;
typedef int32_t int_t;

typedef uint64_t ulong_t;
typedef int64_t long_t;

typedef glm::vec2 vec2_t;

typedef vec2_t uv_t;

typedef glm::vec3 vec3_t;

typedef glm::vec4 vec4_t;

struct tri_t {
	uint_t a, b, c;
};

typedef vec4_t color_t;

typedef glm::mat3 mat3_t;

typedef glm::mat4 mat4_t;

typedef glm::ivec2 ivec2_t;

#endif
