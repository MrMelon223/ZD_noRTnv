#ifndef ZDLEVEL_H
#define ZDLEVEL_H

#include "Runtime.h"
#include "ZDcamera.h"
#include "ZDlight.h"

class ZDlevel {
protected:
	std::string file_path, name, description;

	std::vector<ZDmodel> host_models;
	std::vector<d_ZDmodel> device_models;
	std::vector<d_ZDinstance> device_instances;

	std::vector<ZDtexture> host_textures;
	std::vector<d_ZDtexture> device_textures;

	std::vector<d_ZDpoint_light> device_lights;

	ZDcamera* camera;

	int_t find_host_model(std::string);

	void load_from(std::string);
public:
	ZDlevel(std::string, std::string);

	d_ZDmodel* get_device_models() { return this->device_models.data(); }
	d_ZDinstance* get_instances() { return this->device_instances.data(); }
	size_t get_model_count() { return this->device_models.size(); }
	size_t get_instance_count() { return this->device_instances.size(); }
	size_t get_light_count() { return this->device_lights.size(); }

	ZDcamera* get_camera() { return this->camera; }

	d_ZDmodel* models_to_gpu();
	d_ZDinstance* instances_to_gpu();
	d_ZDtexture* textures_to_gpu();
	d_ZDpoint_light* lights_to_gpu();
};

#endif
