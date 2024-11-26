#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>

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
	const float UI_VIEW_PORTION = 0.2f;
	const float GRID_PADDING = 20.f;
	const float CELL_OUTLINE_PORTION = 0.f;
	// const float CELL_OUTLINE_PORTION = 0.1f;

	// Settings
	float cell_size;
	int grid_width;
	int grid_height;
	
public:
	/* Constructor */
	Visualization(int window_width, int window_height, int grid_width, int grid_height);

	/* Public Methods */
	void process_window_events();
	void draw_grid();
	void draw_ui();
	void clear();
	void display();

	/* Getters */
	bool is_window_open() const;
	float get_ui_view_offset() const;
	float get_ui_view_width() const;
	tgui::Gui& get_gui();

private:
	/* Private Methods */
	void find_grid_dimensions();
	void update_views();
	void handle_mouse_click(int mouse_x, int mouse_y);
};