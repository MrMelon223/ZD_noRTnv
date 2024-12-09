	// main.cpp
#include "../include/ZDapp.h"

int main() {
	ZDapp* app = new ZDapp(1280, 720);

	app->main_loop();

	return 0;
}