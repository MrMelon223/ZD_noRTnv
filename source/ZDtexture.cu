	// ZDtexture.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "../include/ZDtexture.h"

void ZDtexture::load_from(std::string file_path) {
	this->file_path = file_path;

	int w, h, channels_in, desired_channels = 4;

	unsigned char* raw = stbi_load(this->file_path.c_str(), &w, &h, &channels_in, desired_channels);

	this->width = static_cast<int_t>(w);
	this->height = static_cast<int_t>(h);
	this->channels = static_cast<int_t>(channels_in);

	std::cout << std::setw(10) << "Img Dims: { " << this->width << ", " << this->height << ", " << this->channels << " }" << std::endl;

	for (int_t y = 0; y < this->height; y++) {
		for (int_t x = 0; x < this->width; x++) {
			if (this->channels == 1) {
				this->data.push_back(color_t{ static_cast<float>(raw[y * 1 + (x)]) / 255.0f,
					static_cast<float>(raw[y * 1 + (x)]) / 255.0f,
					static_cast<float>(raw[y * 1 + (x)]) / 255.0f,
					static_cast<float>(raw[y * 1 + (x)]) / 255.0f });
			}
			else if (this->channels == 3) {
				this->data.push_back(color_t{ static_cast<float>(raw[y * 3 + (x)]) / 255.0f,
					static_cast<float>(raw[y * 3 + (x + 1)]) / 255.0f,
					static_cast<float>(raw[y * 3 + (x + 2)]) / 255.0f,
					1.0f });
			}
			else if (channels == 4) {
				this->data.push_back(color_t{ static_cast<float>(raw[(y * 4) * this->width + (x)]) / 255.0f,
					static_cast<float>(raw[(y * 4) * this->width + (x+1)]) / 255.0f,
					static_cast<float>(raw[(y * 4) * this->width + (x+2)]) / 255.0f,
					static_cast<float>(raw[(y * 4) * this->width + (x+3)]) / 255.0f });
			}
		}
	}
}

ZDtexture::ZDtexture(std::string path, std::string name) {
	this->name = name;
	this->load_from(path);
}

d_ZDtexture ZDtexture::to_gpu() {
	d_ZDtexture t{ this->width, this->height, nullptr };

	cuda_check(cudaMalloc((void**)&t.data, sizeof(color_t) * this->height * this->width));
	cuda_check(cudaMemcpy(t.data, this->data.data(), this->height * this->width * sizeof(color_t), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	return t;
}

/*__device__
inline color_t d_ZDtexture::sample(float x, float y) {
	if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y < 1.0f) {
		return data[static_cast<int_t>(y * this->height) * this->width + static_cast<int_t>(x * this->width)];
	}
	else {
		return color_t{ 1.0f, 0.0f, 0.0f, 1.0f };
	}
}*/