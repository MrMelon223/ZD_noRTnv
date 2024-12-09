#ifndef ZDAPP_H
#define ZDAPP_H

#include "ZDframebuffer.h"
#include "ZDrender.h"

static void keyboard_callback(GLFWwindow*, int, int, int, int);
static void mouse_callback(GLFWwindow*, int, int, int);

struct KeyboardButtonUse {
	int key, scancode, action, mods;
};
struct MouseButtonUse {
	int button, action, mods;
};

extern std::queue<KeyboardButtonUse> keyboard_button_uses;
extern std::queue<MouseButtonUse> mouse_button_uses;

class ZDapp {
protected:
	int_t width, height;

	d_ZDframebuffer* frame_buffer;
	GLFWwindow* window;

	bool loop;

	ZDlevel* current_level;

	void empty_queues();

	void input_handle(KeyboardButtonUse&);
	void mouse_handle(MouseButtonUse&);

	void gamepad_handle();

	float last_time;
	bool gamepad;

	void setup_gamepad();
public:
	ZDapp(int_t, int_t);

	void main_loop();
};

#endif