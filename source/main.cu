	// main.cpp
#include "../include/ZDapp.h"

int main() {
	ZDapp* app = new ZDapp(1920, 1080);

	app->main_loop();

	return 0;
}