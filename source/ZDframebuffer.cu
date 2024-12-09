	// ZDframebuffer.cpp
#include "../include/ZDframebuffer.h"

void copy_color_buffer(d_ZDframebuffer* fb, color_t* c_buffer) {

	cuda_check(cudaMemcpy(c_buffer, fb->color_buffer, sizeof(color_t) * fb->width * fb->height, cudaMemcpyDeviceToHost));
}

d_ZDframebuffer* create_framebuffer(int_t width, int_t height) {
	d_ZDframebuffer* buff = new d_ZDframebuffer{ nullptr, nullptr, -1, -1 };


	buff->width = width;
	buff->height = height;

	cuda_check(cudaMalloc((void**)&buff->color_buffer, sizeof(color_t) * static_cast<size_t>(buff->width) * buff->height));
	cuda_check(cudaMalloc((void**)&buff->depth_buffer, sizeof(float) * static_cast<size_t>(buff->width) * buff->height));
	cudaDeviceSynchronize();

	zero_buffers(buff);

	return buff;
}

__global__
void zero_buffers(color_t* color_buff, float* depth_buff, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % width,
		y = ((j * 128 + i) - x) / width;
	uint32_t idx = y * width + x;

	color_buff[idx] = color_t(0.0f);
	depth_buff[idx] = 0.0f;
}

void zero_buffers(d_ZDframebuffer* buff) {
	
	color_t* d_color_buff = buff->color_buffer;
	float* d_depth_buff = buff->depth_buffer;

	zero_buffers <<<(buff->height * buff->width) / 128, 128 >> > (d_color_buff, d_depth_buff, buff->width, buff->height);
}

void cleanup_framebuffer(d_ZDframebuffer* buff) {
	cudaFree(buff->color_buffer);
	cudaFree(buff->depth_buffer);
}