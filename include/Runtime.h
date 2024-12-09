#ifndef RUNTIME_H
#define RUNTIME_H

#include "ZDmodel.h"
#include "ZDtexture.h"

namespace ZDruntime {
	const std::string master_model_path = "resources/models.txt";
	const std::string master_texture_path = "resources/textures.txt";

	extern std::vector<ZDmodel> HOST_MODELS;
	extern std::vector<ZDtexture> HOST_TEXTURES;

	int_t find_model_index(std::string);

	void load_models();
	void load_textures();
}

#endif