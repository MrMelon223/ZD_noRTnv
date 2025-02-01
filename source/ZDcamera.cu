	// ZDcamera.cpp
#include "../include/ZDcamera.h"

float controller_look_sensitivity = 0.5f;

ZDcamera::ZDcamera(int_t width, int_t height, float fov, vec3_t position, vec3_t direction) {

	this->width = width;
	this->height = height;

	this->position = position;
	this->direction = glm::normalize(direction);

	this->rotation.x = atan2f(this->direction.y, this->direction.x);
	this->rotation.y = asinf(-this->direction.z);
	this->rotation.z = 0.0f;

	this->hori_fov = fov;
}

d_ZDcamera* ZDcamera::to_gpu() {
	d_ZDcamera cam{ this->position, this->direction, this->rotation, this->hori_fov, nullptr };

	cuda_check(cudaMalloc((void**)&cam.vertex_samples, sizeof(d_ZDvertex_sample) * this->width * this->height));

	d_ZDcamera* d_cam;
	cuda_check(cudaMalloc((void**)&d_cam, sizeof(d_ZDcamera)));
	cuda_check(cudaMemcpy(d_cam, &cam, sizeof(d_ZDcamera), cudaMemcpyHostToDevice));

	return d_cam;
}

d_ZDvertex_sample* from_gpu(d_ZDcamera* cam) {
	d_ZDcamera* d_cam = new d_ZDcamera{};

	cuda_check(cudaMemcpy(d_cam, cam, sizeof(d_ZDcamera), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	return d_cam->vertex_samples;
}

void ZDcamera::update_direction(float x, float y) {
	//printf("X,Y input mouse coord = {%.2f, %.2f}\n", rot.x, rot.y);
	float normalized_coord_x = ((x - (static_cast<float>(this->width) * 0.5f)) / static_cast<float>(this->width));
	float normalized_coord_y = ((y - (static_cast<float>(this->height) * 0.5f)) / static_cast<float>(this->height));
	//printf("X,Y normalized input mouse coord = {%.2f, %.2f}\n", normalized_coord_x, normalized_coord_y);

	float aspect_ratio = static_cast<float>(this->width) / static_cast<float>(this->height);

	float fov_hori_rad = this->hori_fov;
	float fov_vert_rad = this->hori_fov * (static_cast<float>(this->height) / static_cast<float>(this->width));
	float half_fov_hori_rad = fov_hori_rad * 0.5f;
	float half_fov_vert_rad = fov_vert_rad * 0.5f;

	float view_x = normalized_coord_x * half_fov_hori_rad * aspect_ratio;
	float view_y = normalized_coord_y * half_fov_vert_rad;

	this->rotation.y += view_x * 0.01f; //* (static_cast<float>(this->dims.x) / this->dims.y);
	this->rotation.x += view_y * 0.01f;
	this->rotation.z = 0.0f;

	if (this->rotation.x > PI * 0.5f) {
		this->rotation.x = PI * 0.5f;
	}
	if (this->rotation.x < -PI * 0.5f) {
		this->rotation.x = -PI * 0.5f;
	}
	this->rotation.x = fmod(this->rotation.x, 2.0f * PI);
	this->rotation.y = fmod(this->rotation.y, 2.0f * PI);

	float yaw = this->rotation.x * (PI / 180.0f),
		pitch = this->rotation.y * (PI / 180.0f);

	this->direction.x = cosf(yaw) * cosf(pitch);
	this->direction.y = sinf(pitch);
	this->direction.z = sinf(yaw) * cosf(pitch);

	this->direction = glm::normalize(this->direction);
}

void ZDcamera::update_direction() {

	this->rotation.x = fmod(this->rotation.x, 2.0f * PI);
	this->rotation.y = fmod(this->rotation.y, 2.0f * PI);

	if (this->rotation.x > PI * 0.5f) {
		this->rotation.x = PI * 0.5f;
	}
	if (this->rotation.x < -PI * 0.5f) {
		this->rotation.x = -PI * 0.5f;
	}

	float yaw = this->rotation.x,
		pitch = this->rotation.y;

	this->direction.x = cosf(yaw) * cosf(pitch);
	this->direction.y = sinf(pitch);
	this->direction.z = sinf(yaw) * cosf(pitch);

	this->direction = glm::normalize(this->direction);
}

void ZDcamera::forward(float t) {
	this->position = this->position + (glm::normalize(this->direction) * t * 0.5f);

	if (this->position.x < 0.0f) {
		this->position.x = 0.0f;
	}
	if (this->position.y < 0.0f) {
		this->position.y = 0.0f;
	}
	if (this->position.z < 0.0f) {
		this->position.z = 0.0f;
	}
}

void ZDcamera::backward(float t) {
	this->position = this->position - glm::normalize(this->direction) * t * 0.5f;
}
void ZDcamera::right(float t) {
	this->position = this->position - glm::cross(this->direction, vec3_t(0.0f, 1.0f, 0.0f)) * t * 0.5f;
}
void ZDcamera::left(float t) {
	this->position = this->position + glm::cross(this->direction, vec3_t(0.0f, 1.0f, 0.0f)) * t * 0.5f;
}

void ZDcamera::turn_right(float t) {
	float rot = 0.1f * controller_look_sensitivity;
	if (rot >= ZDcamera::DEADZONE) {
		this->rotation.y += rot;
		this->rotation.y = fmod(this->rotation.y, 2.0f * PI);
	}
}
void ZDcamera::turn_left(float t) {
	float rot = 0.1f * controller_look_sensitivity;
	if (rot >= ZDcamera::DEADZONE) {
		this->rotation.y -= rot;
		this->rotation.y = fmod(this->rotation.y, 2.0f * PI);
	}
}

void ZDcamera::turn_up(float t) {
	float rot = 0.1f * controller_look_sensitivity;
	if (rot >= ZDcamera::DEADZONE) {
		this->rotation.x -= rot;
		this->rotation.x = fmod(this->rotation.y, 2.0f * PI);
	}
}
void ZDcamera::turn_down(float t) {
	float rot = 0.1f * controller_look_sensitivity;
	if (rot >= ZDcamera::DEADZONE) {
		this->rotation.x += rot;
		this->rotation.x = fmod(this->rotation.x, 2.0f * PI);
	}
}

void ZDcamera::turn_right_for(float t) {
	t = t / static_cast<float>(this->width);
	if (t >= DEADZONE || t <= -DEADZONE) {
		this->rotation.y += t * 0.1f * controller_look_sensitivity;

		if (this->rotation.y < 0.0f) {
			this->rotation.y = fmod(this->rotation.y, -2.0f * PI);
		}
		else {
			this->rotation.y = fmod(this->rotation.y, 2.0f * PI);
		}

	}
}
void ZDcamera::look_up_for(float t) {
	t = t / static_cast<float>(this->height);
	if (t >= DEADZONE || t <= -DEADZONE) {
		this->rotation.x += t * 0.1f * controller_look_sensitivity;
	}
	if (this->rotation.x < 0.0f) {
		this->rotation.x = fmod(this->rotation.x, -2.0f * PI);
	}
	else {
		this->rotation.x = fmod(this->rotation.x, 2.0f * PI);
	}

	if (this->rotation.x > PI / 2.0f) {
		this->rotation.x = PI / 2.0f;
	}
	if (this->rotation.x < -PI / 2.0f) {
		this->rotation.x = -PI / 2.0f;
	}
}

void ZDcamera::debug_print() {
	std::cout << "Camera Debug:" << std::endl;
	std::cout << std::setw(15) << "Res: " << this->width << "x" << this->height << "p" << std::endl;
	std::cout << std::setw(15) << "Loc: { " << this->position.x << ", " << this->position.y << ", " << this->position.z << " }" << std::endl;
	std::cout << std::setw(15) << "Dir: { " << this->direction.x << ", " << this->direction.y << ", " << this->direction.z << " }" << std::endl;
}