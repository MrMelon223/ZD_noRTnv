	// ZDrender.cpp
#include "../include/ZDrender.h"

__device__
float line_equation(vec2_t p, vec2_t p1, vec2_t p2) {
	float A = p2.y - p1.y;
	float B = p1.x - p2.x;
	float C = p2.x * p1.y - p1.x * p2.y;
	return A * p.x + B * p.y + C;
}

__device__
float tri_area(vec2_t a, vec2_t b, vec2_t c) {
	vec2_t va = b - a,
		vb = b - c;
	float vc = va.x * vb.y - va.y * vb.x;
	return vc / 2.0f;
}

__device__
uv_t compute_barycentric(vec2_t p, uv_t a, uv_t b, uv_t c) {
	float denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);

	// Calculate barycentric coordinates lambdaA, lambdaB, lambdaC
	//float lambdaA = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
	//float lambdaB = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
	//float lambdaC = 1.0f - lambdaA - lambdaB;

	uv_t v0 = b - a,
		v1 = c - a,
		v2 = p - a;
	/*float d00 = v0.x * v0.y - v0.x * v0.y;
	float d01 = v0.x * v1.y - v1.x * v0.y;
	float d11 = v1.x * v1.y - v1.x * v1.y;
	float d20 = v2.x * v0.y - v0.x * v2.y;
	float d21 = v2.x * v1.y - v2.x * v1.y;
	float denom = d00 * d11 - d01 * d01;*/

	//float area_abc = tri_area(a, b, c);
	//float area_pbc = tri_area(p, b, c);
	//float area_pca = tri_area(p, c, a);

	// The point's barycentric coordinates (lambdaA, lambdaB, lambdaC) relative to the triangle
	uv_t barycentric;

	barycentric.x = ((b.y - c.y)*(p.x - c.x) + (c.x - b.x)*(a.y - c.y)) / denom;
	barycentric.y = ((c.y - a.y)*(p.x - c.x) + (a.x - c.x)*(p.y - c.y)) / denom;

	//barycentric.x = area_pbc / area_abc; //lambdaA;  // Corresponds to the weight for vertex A
	//barycentric.y = area_pca / area_abc; //lambdaB;  // Corresponds to the weight for vertex B
	//barycentric.x = (d11 * d20 - d01 * d21) / denom;
	//barycentric.y = (d00 * d21 - d01 * d20) / denom;
	// You could return lambdaC as well, but since the sum is 1, we skip it here.

	return barycentric;
}

__global__
void calculate_visibility(d_ZDmodel* d_models, d_ZDinstance* d_instances, int_t instance_count, d_ZDcamera* d_camera) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t index = j * 128 + i;

	if (index < instance_count) {
		d_ZDinstance* inst = &d_instances[index];
		d_ZDmodel* model = &d_models[inst->model_index];

		vec3_t cam_to_inst = d_camera->position - inst->position;
		float dist = sqrtf(cam_to_inst.x * cam_to_inst.x + cam_to_inst.y * cam_to_inst.y + cam_to_inst.z * cam_to_inst.z);
		float d_prod = glm::dot(cam_to_inst, d_camera->direction);
		if (d_prod <= 0.0f || dist < 0.01f || dist > 1000.0f) {
			inst->show = false;
		}
		else {
			cam_to_inst = glm::normalize(cam_to_inst);
			float angle = glm::dot(d_camera->direction, cam_to_inst);
			if (angle >= cosf(d_camera->hori_fov * 0.5f)) {
				inst->show = true;
			}
			else {
				inst->show = false;
			}
		}
		//inst->show = true;
	}
}

void ZDrender::calculate_instance_visibility(d_ZDmodel* models, d_ZDinstance* instances, int_t instance_count, d_ZDcamera* camera) {
	calculate_visibility << <(instance_count / 128) + 1, 128 >> > (models, instances, instance_count, camera);
}

__global__
void transform_vertices(d_ZDmodel* models, d_ZDinstance* instances, d_ZDcamera* camera, int_t instance_count, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t index = j * 128 + i;

	if (index < instance_count) {
		d_ZDinstance* inst = &instances[index];	
		d_ZDmodel* model = &models[inst->model_index];

		vec3_t* positions = model->vertex_positions;
		vec3_t* normals = model->triangle_normals;
		tri_t* tri_idxs = model->triangle_indices;
		// float fov, float aspectRatio, float nearPlane, float farPlane
		mat4_t perspect = glm::perspective(camera->hori_fov, static_cast<float>(width) / height, 0.1f, 100.0f);
		bool fail_invert = false;

		mat4_t x_rot = glm::rotate(mat4_t(1.0f), camera->rotation.x, vec3_t(1.0f, 0.0f, 0.0f));
		mat4_t y_rot = glm::rotate(mat4_t(1.0f), camera->rotation.y, vec3_t(0.0f, 1.0f, 0.0f));
		mat4_t z_rot = glm::rotate(mat4_t(1.0f), camera->rotation.z, vec3_t(0.0f, 0.0f, 1.0f));

		mat4_t cam_mtx = glm::inverse(x_rot * y_rot * z_rot);

		mat4_t rotation = x_rot * y_rot * z_rot;

		for (uint_t i = 0; i < inst->vertex_count; i++) {
			//if (inst->visible_triangles[i]) {
			vec3_t v0 = positions[i];
			v0 = (camera->position - v0);

			vec4_t v = rotation * vec4_t(v0.x, v0.y, v0.z, 0.0f);
			v = cam_mtx * v;
			v = perspect * v;

			v0 = vec3_t(v.x, v.y, v.z);


			inst->transformed_vertices[i] = vec3_t{ inst->scale * v0.x, inst->scale * v0.y, inst->scale * v0.z };
		}
		for (uint_t i = 0; i < inst->triangle_count; i++) {
			vec3_t v0 = normals[i];
			//v0 = ZD::add_v3(v0, ZD::subtract_v3(d_camera->position, inst->position));

			vec4_t n_t = rotation * vec4_t(v0.x, v0.y, v0.z, 0.0f);
			//v0 = ZD::to_vec3(ZD::product_m4(cam_mtx, ZD::to_vec4(v0, 0.0f)));
			v0 = vec3_t(n_t.x, n_t.y, n_t.z);

			inst->transformed_normals[i] = v0;
		}
	}
}

__global__
void draw_instances(color_t* color_buff, float* depth_buff, d_ZDmodel* models, d_ZDinstance* instances, int_t instance_count, d_ZDcamera* camera, d_ZDtexture* textures, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t index = j * 128 + i;
	uint_t x = index % width,
		y = (index - x) / width;
	if (index < instance_count) {
		d_ZDinstance* inst = &instances[index];
		d_ZDmodel* model = &models[inst->model_index];

		vec3_t* positions = model->vertex_positions;
		vec3_t* normals = model->triangle_normals;
		tri_t* tri_idxs = model->triangle_indices;
		// float fov, float aspectRatio, float nearPlane, float farPlane
		mat4_t perspect = glm::perspective(camera->hori_fov, static_cast<float>(width) / height, 0.1f, 100.0f);
		bool fail_invert = false;

		mat4_t cam_mtx_a = glm::inverse(glm::rotate(mat4_t(1.0f), camera->rotation.x, vec3_t(1.0f, 0.0f, 0.0f)));
		mat4_t cam_mtx_y = glm::inverse(glm::rotate(mat4_t(1.0f), camera->rotation.y, vec3_t(0.0f, 1.0f, 0.0f)));
		mat4_t cam_mtx_z = glm::inverse(glm::rotate(mat4_t(1.0f), camera->rotation.z, vec3_t(0.0f, 0.0f, 1.0f)));
		mat4_t cam_mtx = cam_mtx_a * cam_mtx_y * cam_mtx_z;

		mat4_t rotation_x = glm::rotate(mat4_t(1.0f), inst->rotation.x, vec3_t(1.0f, 0.0f, 0.0f));
		mat4_t rotation_y = glm::rotate(mat4_t(1.0f), inst->rotation.y, vec3_t(0.0f, 1.0f, 0.0f));
		mat4_t rotation_z = glm::rotate(mat4_t(1.0f), inst->rotation.z, vec3_t(0.0f, 0.0f, 1.0f));
		mat4_t rotation = rotation_x * rotation_y * rotation_z;

		for (uint_t i = 0; i < inst->vertex_count; i++) {
			//if (inst->visible_triangles[i]) {
			vec4_t v0 = vec4_t(positions[i].x, positions[i].y, positions[i].z, 0.0f);
			v0 = v0 + vec4_t(camera->position - inst->position, 0.0f);

			v0 = rotation * v0;
			v0 = cam_mtx * v0;
			v0 = perspect * v0;


			inst->transformed_vertices[i] = vec3_t{ inst->scale * v0.x, inst->scale * v0.y, inst->scale * v0.z };
		}
		for (uint_t i = 0; i < inst->triangle_count; i++) {
			vec4_t v0 = vec4_t(normals[i].x, normals[i].y, normals[i].z, 0.0f);
			//v0 = ZD::add_v3(v0, ZD::subtract_v3(d_camera->position, inst->position));

			v0 = rotation * v0;
			//v0 = ZD::to_vec3(ZD::product_m4(cam_mtx, ZD::to_vec4(v0, 0.0f)));


			inst->transformed_normals[i] = vec3_t(v0.x, v0.y, v0.z);
		}
	}
}

__global__
void interpolate_instances(color_t* color_buff, float* depth_buff, d_ZDmodel* models, d_ZDinstance* instances, int_t instance_count, d_ZDcamera* camera, d_ZDtexture* textures, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t index = j * 128 + i;

	int_t x = index % width,
		y = (index - x) / width;
	vec2_t normalized_coord{};

	float ratio = static_cast<float>(width) / static_cast<float>(height);
	float norm_x = (static_cast<float>(x) - (static_cast<float>(width) * 0.5f)) / (static_cast<float>(width));
	float norm_y = (static_cast<float>(y) - (static_cast<float>(height) * 0.5f)) / (static_cast<float>(height));
	float fov_rad = camera->hori_fov * (PI / 180.0f);
	float half_fov = fov_rad * 0.5f;

	normalized_coord.x = norm_x;
	normalized_coord.y = norm_y;

	color_buff[y * width + x] = color_t{ 0.17f, 0.15f, 0.17f, 1.0f };
	d_ZDvertex_sample* samp = &camera->vertex_samples[y * width + x];
	samp->depth = 1000.0f;
	samp->hit = false;

	for (uint_t i = 0; i < instance_count; i++) {
		d_ZDinstance* instance = &instances[i];
		if (instance->show) {
			d_ZDmodel* model = &models[instance->model_index];
			for (uint_t j = 0; j < instance->triangle_count; j++) {
				//if (instance->visible_triangles[j]) {
				tri_t t = model->triangle_indices[j];
				vec3_t normal = instance->transformed_normals[j];

				//float n_dot = ZD::dot(normal, d_camera->direction);
				//if (n_dot <= 0.0f) {

				vec3_t v0 = instance->transformed_vertices[t.a],
					v1 = instance->transformed_vertices[t.b],
					v2 = instance->transformed_vertices[t.c];

				vec3_t min = glm::max(v0, glm::max(v1, v2));

				vec3_t depth = camera->position - min;
				float depth_test = min.z;

				if (depth_test < samp->depth && depth_test > 0.0f) {
					vec2_t v0a = { -v0.x / v0.z, -v0.y / v0.z },
						v1a = { -v1.x / v1.z, -v1.y / v1.z },
						v2a = { -v2.x / v2.z, -v2.y / v2.z };

					float min_x = glm::min(v0a.x, glm::min(v1a.x, v2a.x));
					float max_x = glm::max(v0a.x, glm::max(v1a.x, v2a.x));

					float min_y = glm::min(v0a.y, glm::min(v1a.y, v2a.y));
					float max_y = glm::max(v0a.y, glm::max(v1a.y, v2a.y));

					if ((max_x > -1.0f || min_x < 1.0f) && (min_x > -1.0f && max_x < 1.0f) && (min_y > -1.0f || max_y < 1.0f) && (min_y > -1.0f && max_y < 1.0f)/* && !((min_x < -1.0f && max_x > 1.0f) || (min_y < -1.0f && max_y > 1.0f))*/) {
						float sign1 = line_equation(normalized_coord, v0a, v1a),
							sign2 = line_equation(normalized_coord, v1a, v2a),
							sign3 = line_equation(normalized_coord, v2a, v0a);


						if ((sign1 >= 0.0f && sign2 >= 0.0f && sign3 >= 0.0f) ||
							(sign1 <= 0.0f && sign2 <= 0.0f && sign3 <= 0.0f)) {
							uv_t min = { glm::min(v0a.x, glm::min(v1a.x, v2a.x)), glm::min(v0a.y, glm::min(v1a.y, v2a.y)) };
							uv_t max = { glm::max(v0a.x, glm::max(v1a.x, v2a.x)), glm::max(v0a.y, glm::max(v1a.y, v2a.y)) };
							uv_t comp_coord = uv_t{ (static_cast<float>(x) - min.x) / (max.x - min.x), (static_cast<float>(y) - min.y) / (max.y - min.y) };


							uv_t uv = compute_barycentric(comp_coord, models[instance->model_index].vertex_uvs[t.a], models[instance->model_index].vertex_uvs[t.b], models[instance->model_index].vertex_uvs[t.c]);

							samp->instance_index = i;
							samp->uv_coord = uv;
							samp->hit = true;

							//d_color_buff[y * *d_width + x] = color_t{ depth_test / 100.0f, depth_test / 100.0f, depth_test / 100.0f, 1.0f };
							samp->depth = depth_test;
							samp->model_index = instance->model_index;
							samp->triangle_index = j;
							samp->triangle_normal = normal;

							float ratio = static_cast<float>(width) / static_cast<float>(height);
							float norm_x = (x - (static_cast<float>(width) * 0.5f)) / (static_cast<float>(width) * 0.5f);
							float norm_y = (y - (static_cast<float>(height) * 0.5f)) / (static_cast<float>(height) * 0.5f);
							float fov_rad = camera->hori_fov * (PI / 180.0f);
							float half_fov = fov_rad * 0.5f;

							vec3_t upward{ 0.0f, 1.0f, 0.0f };

							vec3_t right = glm::cross(camera->direction, upward);

							vec3_t up = glm::cross(right, camera->direction);
							up = glm::normalize(up);
							vec3_t direction = camera->direction;
							right = glm::normalize(right);

							direction.x = direction.x + norm_x * half_fov * ratio * right.x + norm_y * half_fov * up.x;
							direction.y = direction.y + norm_x * half_fov * ratio * right.y + norm_y * half_fov * up.y;
							direction.z = direction.z + norm_x * half_fov * ratio * right.z + norm_y * half_fov * up.z;

							samp->position = camera->position + direction * depth_test;
							depth_buff[y * width + x] = depth_test;
						}
						else {
						}
					}
					else {

					}
				}
			}
		}
	}
}

__global__
void flat_shade(color_t* color_buff, float* depth_buff, d_ZDmodel* models, d_ZDinstance* instances, int_t instance_count, d_ZDcamera* camera, d_ZDtexture* textures, d_ZDpoint_light* point_lights, int_t point_light_count, d_ZDambient_light* ambient_light, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t index = j * 128 + i;
	int_t x = index % width,
		y = (index - x) / width;

	d_ZDvertex_sample* smp = &camera->vertex_samples[y * width + x];
	if (smp->hit) {
		//vec3_t sun_direction = vec3_t{ 0.10f, 0.33f, 0.2f };
		//color_t sun_color = color_t{ 0.72f, 0.45f, 0.93f, 1.0f };
		float sun_intensity = 100.0f;
		float cum_intensity = 0.0f;

		color_t t_samp = textures[instances[smp->instance_index].diffuse_index].sample(smp->uv_coord.x, smp->uv_coord.y);

		/*float diffuse = glm::dot(smp->triangle_normal, sun_direction);
		if (diffuse >= 0.0f) {
			//d_color_buff[index] = color_t{ sun_intensity * sun_color.x * diffuse, sun_intensity * sun_color.y * diffuse, sun_intensity * sun_color.z * diffuse , sun_intensity * sun_color.w * diffuse };
			//d_color_buff[y * *d_width + x] = color_t{ 1.0f, 1.0f, 1.0f, 1.0f };
			color_buff[y * width + x] = color_buff[y * width + x] + t_samp;
		}
		else {
			color_buff[y * width + x] = color_buff[y * width + x] + t_samp;
		}*/
		color_t pixel(0.0f);
		for (int_t k = 0; k < point_light_count; k++) {
			vec3_t to_light = point_lights[k].position - smp->position;

			float mag = glm::length(to_light);

			float intensity = point_lights[k].intensity / (mag * mag * point_lights[k].falloff_distance);
			float bright = glm::dot(to_light, smp->triangle_normal);
			//if (bright >= 0.0f) {
				pixel += point_lights[k].diffuse_color;
			//}
			cum_intensity += /*bright */ intensity;

		}
		color_buff[y * width + x] = (cum_intensity * t_samp) + (cum_intensity * pixel);
	}
}

void ZDrender::draw(d_ZDframebuffer* buff, d_ZDmodel* models, d_ZDtexture* textures, d_ZDinstance* instances, d_ZDcamera* camera, int_t instance_count, d_ZDpoint_light* point_lights, int_t light_count, d_ZDambient_light* ambient_light) {
	color_t* d_color_buff = buff->color_buffer;
	float* d_depth_buff = buff->depth_buffer;

	d_ZDtexture* d_textures = textures;
	
	/*std::cout << "Pointers: " << std::endl;
	std::cout << std::setw(10) << models << std::endl;
	std::cout << std::setw(10) << instances << std::endl;
	std::cout << std::setw(10) << camera << std::endl;
	std::cout << std::setw(10) << instance_count << std::endl;
	std::cout << std::setw(10) << buff->width << std::endl;
	std::cout << std::setw(10) << buff->height << std::endl;*/

	transform_vertices << < (instance_count / 128) + 1, 128 >> > (models, instances, camera, instance_count, buff->width, buff->height);
	cudaDeviceSynchronize();
	draw_instances <<< (instance_count / 128) + 1, 128 >>> (d_color_buff, d_depth_buff, models, instances, instance_count, camera, textures, buff->width, buff->height);
	cudaDeviceSynchronize();
	interpolate_instances << < (buff->height * buff->width) / 128, 128 >> > (d_color_buff, d_depth_buff, models, instances, instance_count, camera, textures, buff->width, buff->height);
	cudaDeviceSynchronize();
	flat_shade << < (buff->height * buff->width) / 128, 128 >> > (d_color_buff, d_depth_buff, models, instances, instance_count, camera, textures, point_lights, light_count, ambient_light, buff->width, buff->height);
	cudaDeviceSynchronize();
	cuda_check(cudaGetLastError(), "CU_FLAT_SHADE::ERROR");
	//cuda_check(cudaGetLastError(), "CURENDER::ERROR");
}