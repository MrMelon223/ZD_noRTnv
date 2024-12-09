	// Runtime.cpp
#include "../include/Runtime.h"

	// Runtime Variables
std::vector<ZDmodel> ZDruntime::HOST_MODELS;
std::vector<ZDtexture> ZDruntime::HOST_TEXTURES;

void cuda_check(cudaError_t e) {
	if (e != cudaSuccess) {
		std::cerr << cudaGetErrorString(e) << std::endl;
	}
}

void cuda_check(cudaError_t e, std::string line) {
	if (e != cudaSuccess) {
		std::cerr << line << ": " << cudaGetErrorString(e) << std::endl;
	}
}

int_t ZDruntime::find_model_index(std::string name) {
	int_t r;
	for (r = 0; r < static_cast<int_t>(ZDruntime::HOST_MODELS.size()); r++) {
		if (ZDruntime::HOST_MODELS[r].get_name() == name) {
			return r;
		}
	}
	std::cerr << "Cannot find ZDmodel: " << name << std::endl;
	return -1;
}

void ZDruntime::load_models() {
	std::ifstream in(ZDruntime::master_model_path, std::ios::in);

	if (!in) {
		std::cerr << "Cannot Find Model List File!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(in, line)) {
		std::istringstream line_in(line);
		std::string path = line.substr(0, line.find(","));
		std::string name = line.substr(line.find(",") + 1, line.size());

		printf("Path:Name = %s : %s\n", path.c_str(), name.c_str());

		ZDruntime::HOST_MODELS.push_back(ZDmodel(path, name));
	}
	in.close();
}

void ZDruntime::load_textures() {
	std::ifstream in(ZDruntime::master_texture_path, std::ios::in);

	if (!in) {
		std::cerr << "Cannot Find Texture List File!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(in, line)) {
		std::istringstream line_in(line);
		std::string path = line.substr(0, line.find(","));
		std::string name = line.substr(line.find(",") + 1, line.size());

		printf("Path:Name = %s : %s\n", path.c_str(), name.c_str());

		ZDruntime::HOST_TEXTURES.push_back(ZDtexture(path, name));
	}
	in.close();
}