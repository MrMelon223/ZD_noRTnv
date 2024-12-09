	// ZDlevel.cpp
#include "../include/ZDlevel.h"

int_t ZDlevel::find_host_model(std::string name) {
	int_t r;
	for (r = 0; r < static_cast<int_t>(this->host_models.size()); r++) {
		if (this->host_models[r].get_name() == name) {
			return r;
		}
	}
	std::cerr << "Cannot find ZDmodel: " << name << std::endl;
	return -1;
}

void ZDlevel::load_from(std::string path) {
	this->file_path = path;

	std::ifstream in;
	in.open(path, std::ios::in);
	if (!in) {
		std::cout << "Cannot find Level: " << path << std::endl;
		return;
	}

	std::cout << "Loading Level: " << path << std::endl;

	std::string line;
	std::getline(in, line);
	std::istringstream parse(line);
	size_t leng = 0;

	parse >> leng;
	std::cout << leng << " static models detected!" << std::endl;

	this->host_models = std::vector<ZDmodel>();
	this->device_models = std::vector<d_ZDmodel>();
	this->host_textures = std::vector<ZDtexture>();
	this->device_textures = std::vector<d_ZDtexture>();

	this->host_textures.push_back(ZDtexture("resources/textures/checkerboard.png", "test"));

	for (size_t i = 0; i < leng; i++) {
		std::getline(in, line);
		std::istringstream in0(line);

		float x, y, z, x_r, y_r, z_r, scale;
		std::string model;

		in0 >> x >> y >> z >> x_r >> y_r >> z_r >> scale >> model;
		//std::cout << model << std::endl;

		vec3_t position = vec3_t{ x, y, z };
		vec3_t rotation = vec3_t{ x_r, y_r, z_r };

		int_t h_idx = this->find_host_model(model);

		if (h_idx == -1) {
			int_t found_idx = ZDruntime::find_model_index(model);
			if (found_idx >= 0) {
				this->host_models.push_back(ZDruntime::HOST_MODELS[found_idx]);
				this->device_models.push_back(ZDruntime::HOST_MODELS[found_idx].to_gpu());
				h_idx = static_cast<int_t>(this->host_models.size() - 1);
				std::cout << "Adding ZDmodel: " << ZDruntime::HOST_MODELS[found_idx].get_name() << " to ZDlevel Host Model Vector @ " << h_idx << std::endl;
			}
			else {
				std::cerr << "Cannot Find Model: " << model << " in ZDruntime." << std::endl;
				continue;
			}
		}
		this->device_instances.push_back(create_instance(h_idx, position, rotation, this->host_models[h_idx].get_vertex_count(), this->host_models[h_idx].get_triangle_count(), true, scale, 0));

		std::cout << "d_model = " << this->device_instances.back().model_index << std::endl;
	}


	std::string light_count;
	std::getline(in, light_count);
	std::istringstream count_in(light_count);
	size_t l_count = 0;

	count_in >> l_count;
	std::string line2;
	for (size_t i = 0; i < l_count; i++) {
		std::getline(in, line2);
		std::istringstream in1(line2);

		float x, y, z, r, g, b, a, rs, gs, bs, as, intensity, falloff, range;

		in1 >> x >> y >> z >> r >> g >> b >> a >> rs >> gs >> bs >> as >> intensity >> falloff >> range;

		this->device_lights.push_back(d_ZDpoint_light{ vec3_t(x,y,z), color_t(r,g,b,a), color_t(rs,gs,bs,as), intensity, falloff, range });

	}
}

ZDlevel::ZDlevel(std::string file_path, std::string name) {
	this->name = name;
	this->load_from(file_path);

	this->camera = new ZDcamera(1280, 720, 120.0f, vec3_t{ 100.0f, 100.0f, 100.0f }, vec3_t{ 0.0f, 0.0f, -1.0f });
}

d_ZDmodel* ZDlevel::models_to_gpu() {
	d_ZDmodel* models;
	cuda_check(cudaMalloc((void**)&models, sizeof(d_ZDmodel) * this->device_models.size()));
	cuda_check(cudaMemcpy(models, this->device_models.data(), sizeof(d_ZDmodel) * this->device_models.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	return models;
}

d_ZDinstance* ZDlevel::instances_to_gpu() {
	d_ZDinstance* instances;
	cuda_check(cudaMalloc((void**)&instances, sizeof(d_ZDinstance) * this->device_instances.size()));
	cuda_check(cudaMemcpy(instances, this->device_instances.data(), sizeof(d_ZDinstance) * this->device_instances.size(), cudaMemcpyHostToDevice), "CUMODELTOGPU::ERROR");
	cudaDeviceSynchronize();

	return instances;
}

d_ZDtexture* ZDlevel::textures_to_gpu() {

	for (uint_t i = 0; i < this->host_textures.size(); i++) {
		this->device_textures.push_back(this->host_textures[i].to_gpu());
	}

	d_ZDtexture* textures;
	cuda_check(cudaMalloc((void**)&textures, sizeof(d_ZDtexture) * this->device_textures.size()));
	cuda_check(cudaMemcpy(textures, this->device_textures.data(), sizeof(d_ZDtexture) * this->device_textures.size(), cudaMemcpyHostToDevice), "CUTEXTURETOGPU::ERROR");
	cudaDeviceSynchronize();

	return textures;
}

d_ZDpoint_light* ZDlevel::lights_to_gpu() {
	d_ZDpoint_light* lights;
	cuda_check(cudaMalloc((void**)&lights, sizeof(d_ZDpoint_light) * this->device_lights.size()));
	cuda_check(cudaMemcpy(lights, this->device_lights.data(), sizeof(d_ZDpoint_light) * this->device_lights.size(), cudaMemcpyHostToDevice), "CULIGHTSTOGPU::ERROR");
	cudaDeviceSynchronize();

	return lights;
}