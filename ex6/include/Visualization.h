#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <cstdint>
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

	// Settings
	float cell_size;
	int grid_width;
	int grid_height;
	std::function<void(int, int)> cell_click_callback;

	// Precomputed
	sf::VertexArray grid_vertices;
	sf::VertexArray grid_lines;
	sf::RectangleShape cell_shape;
	sf::RectangleShape grid_background;
	sf::RectangleShape ui_background;

	// Cells from previous frame
	uint16_t* previous_cells;
	bool first_iteration;
	
public:
	/* Constructor */
	Visualization(int window_width, int window_height, int grid_width, int grid_height);
	~Visualization();

	/* Public Methods */
	void process_window_events();
	void init_grid();
	void manage_grid_update(uint16_t* cells, bool force_full_update = false);
	void update_grid_cell(uint16_t* cells, int cell_x, int cell_y);
	void draw_grid(bool draw_grid_lines);
	void init_ui();
	void draw_ui();
	void clear();
	void display();

	/* Getters */
	bool is_window_open() const;
	float get_ui_view_offset() const;
	float get_ui_view_width() const;
	tgui::Gui& get_gui();

	/* Setters */
	void set_cell_click_callback(std::function<void(int, int)> callback);

private:
	/* Private Methods */
	void update_whole_grid(uint16_t* cells);
	void update_grid_cells(uint16_t* cells);
	void find_grid_dimensions();
	void update_views();
	void handle_mouse_click(int mouse_x, int mouse_y);
	sf::Color state_to_color(Automaton::State state);
};