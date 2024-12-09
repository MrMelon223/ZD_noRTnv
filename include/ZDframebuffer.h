#ifndef ZDFRAMEBUFFER_H
#define ZDFRAMEBUFFER_H

#include "Vectors.h"

struct d_ZDframebuffer {
	color_t* color_buffer;
	float* depth_buffer;
	int_t width, height;
};

d_ZDframebuffer* create_framebuffer(int_t, int_t);

void zero_buffers(d_ZDframebuffer*);

void copy_color_buffer(d_ZDframebuffer*, color_t*);

void cleanup_framebuffer(d_ZDframebuffer*);

#endif