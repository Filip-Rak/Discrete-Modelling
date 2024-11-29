#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include "Automaton.h"

class Visualization
{
private:
	/* Attributes */

	// Components
	sf::RenderWindow window;
	tgui::Gui gui;

	// Views
	sf::View grid_view;
	sf::View ui_view;

	// Constants
	const std::map<Automaton::State, sf::Color> state_colors =
	{
		{Automaton::WALL, sf::Color::Black},
		{Automaton::GAS, sf::Color::Blue},
		{Automaton::EMPTY, sf::Color::White}
	};
	const float UI_VIEW_PORTION = 0.2f;
	const float GRID_PADDING = 20.f;
	bool outline_enabled = false;
	float cell_outline_portion = 0.f;
	const float CELL_OUTLINE_PORTION_ENABLED = 0.1f;

	// Settings
	float cell_size;
	int grid_width;
	int grid_height;

	std::function<void(int, int)> cell_click_callback;
	
public:
	/* Constructor */
	Visualization(int window_width, int window_height, int grid_width, int grid_height);

	/* Public Methods */
	void process_window_events();
	void draw_grid(uint16_t* cells);
	void draw_ui();
	void clear();
	void display();
	bool toggle_grid_outline();

	/* Getters */
	bool is_window_open() const;
	float get_ui_view_offset() const;
	float get_ui_view_width() const;
	tgui::Gui& get_gui();

	/* Setters */
	void set_cell_click_callback(std::function<void(int, int)> callback);
	void set_cell_outline(float value);

private:
	/* Private Methods */
	void find_grid_dimensions();
	void update_views();
	void handle_mouse_click(int mouse_x, int mouse_y);
	sf::Color state_to_color(Automaton::State state);
};