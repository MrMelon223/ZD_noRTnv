#ifndef ZDLIGHT_H
#define ZDLIGHT_H

#include "Vectors.h"

struct d_ZDpoint_light {
	vec3_t position;
	color_t diffuse_color;
	color_t specular_color;
	float intensity, falloff_distance, range;
};

struct d_ZDambient_light {
	color_t diffuse_color;
	color_t specular_color;
	float intensity;
};

#endif
