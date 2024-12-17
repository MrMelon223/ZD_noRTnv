#ifndef ZDTEXTURE_H
#define ZDTEXTURE_H

#include "Vectors.h"

struct d_ZDtexture;

class ZDtexture {
protected:
	std::string name;
	std::string file_path;

	int_t width, height, channels;
	std::vector<color_t> data;

	void load_from(std::string);
public:
	ZDtexture(std::string, std::string);
	 
	d_ZDtexture to_gpu();
};

struct d_ZDtexture {
	int_t width, height;
	color_t* data;

	__device__ inline color_t sample(float x, float y) {
		if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
			return data[static_cast<int_t>(y * this->height) * this->width + static_cast<int_t>(x * this->width)];
		}
		else {
			return data[static_cast<int_t>(fmodf(y, 1.0f) * this->height) * this->width + static_cast<int_t>(fmodf(x, 1.0f) * this->width)];
		}
	}
};

#endif