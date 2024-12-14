#ifndef ZDCAMERA_H
#define ZDCAMERA_H

#include "ZDframebuffer.h"

extern float controller_look_sensitivity;

struct d_ZDcamera;
struct d_ZDfrustrum;

struct d_ZDvertex_sample {
	bool hit;
	uint_t model_index,
		triangle_index;
	uint_t instance_index;
	vec3_t triangle_normal;
	vec3_t position;
	float depth;
	uv_t uv_coord;
};

class ZDcamera {
protected:
	int_t width, height;
	vec3_t position, direction, rotation;

	const float DEADZONE = 0.055f;

	float hori_fov;
public:
	ZDcamera(int_t, int_t, float, vec3_t, vec3_t);

	void set_direction(vec3_t d) { this->direction = d; }

	vec3_t get_position() { return this->position; }

	void update_direction(float, float);
	void update_direction();

	void forward(float);
	void backward(float);
	void right(float);
	void left(float);

	void turn_right(float);
	void turn_left(float);
	void turn_up(float);
	void turn_down(float);

	void turn_right_for(float);
	void look_up_for(float);


	void debug_print();

	d_ZDcamera* to_gpu();
};

d_ZDvertex_sample* from_gpu(d_ZDcamera*);

struct d_ZDcamera {
	vec3_t position, direction, rotation;
	float hori_fov;
	d_ZDvertex_sample* vertex_samples;
};

#endif