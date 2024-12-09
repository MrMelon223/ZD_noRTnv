#ifndef ZDMODEL_H
#define ZDMODEL_H

#include "Vectors.h"

struct d_ZDmodel;
struct d_ZDinstance; 

class ZDmodel {
protected:
	std::string file_path;
	std::string name;

	std::vector<vec3_t> vertex_positions;
	std::vector<vec3_t> vertex_normals;
	std::vector<uv_t> vertex_uvs;
	std::vector<color_t> vertex_colors;

	std::vector<tri_t> triangle_indices;
	std::vector<vec3_t> triangle_normals;

	void load_from(std::string);
public:
	ZDmodel(std::string, std::string);

	d_ZDmodel to_gpu();

	std::string get_name() { return this->name; }

	uint_t get_triangle_count() { return static_cast<uint_t>(this->triangle_indices.size()); }
	uint_t get_vertex_count() { return static_cast<uint_t>(this->vertex_positions.size()); }
};

struct d_ZDmodel {

	vec3_t* vertex_positions;
	vec3_t* vertex_normals;
	uv_t* vertex_uvs;
	color_t* vertex_colors;
	uint_t* vertex_count;

	tri_t* triangle_indices;
	vec3_t* triangle_normals;
	uint_t* triangle_count;
};

struct d_ZDinstance {
	int_t model_index;
	vec3_t position;
	vec3_t rotation;
	uint_t vertex_count;
	uint_t triangle_count;
	bool show;
	float scale;
	bool* visible_triangles;
	vec3_t* transformed_vertices;
	vec3_t* transformed_normals;

	uint_t diffuse_index;
};

d_ZDinstance create_instance(int_t, vec3_t, vec3_t, uint_t, uint_t, bool, float, uint_t);
void free_instance(d_ZDinstance*);

#endif