#ifndef HELPER_H
#define HELPER_H

#pragma comment(lib, "opengl32.lib")

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <float.h>
#include <stb_image.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <GLFW/glfw3.h>

__device__
const float PI = 3.1415f;

void cuda_check(cudaError_t);
void cuda_check(cudaError_t, std::string);

#endif