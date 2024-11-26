#include "UI.h"

/* Constructor */
UI::UI(tgui::Gui& gui) : gui_ref(gui) {}

/* Public Methods */
void UI::initialize(float ui_offset_x, float ui_width)
{
	// Pause Button
	auto pause_button = tgui::Button::create("Start");
	pause_button->setSize(80, 40);
	pause_button->setPosition(ui_offset_x + 10, 10);
	pause_button->onPress([this]()
		{
			if (on_pause) 
				on_pause();
		});

	gui_ref.add(pause_button);

	// Reset Button
	auto reset_button = tgui::Button::create("Reset");
	reset_button->setSize(80, 40);
	reset_button->setPosition(ui_offset_x + 10, 100);
	reset_button->onPress([this]()
		{
			if (on_reset)
				on_reset();
		});

	gui_ref.add(reset_button);
}

/* Setters */
void UI::set_pause_callback(const std::function<void()>& callback)
{
	on_pause = callback;
}

void UI::set_reset_callback(const std::function<void()>& callback)
{
	on_reset = callback;
}
