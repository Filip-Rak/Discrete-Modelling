#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <cstdint>
#include "Automaton.h"	// Replace with grid
#include "Grid.h"

class Visualization
{
public:
	/* Enums */
	enum CellVisualState
	{
		EMPTY,
		GAS, 
		WALL
	};

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

	// Settings
	float cell_size;
	int grid_width;
	int grid_height;
	std::function<void(int, int)> cell_click_callback;

	// Colours
	const sf::Color EMPTY_CELL_COLOR = sf::Color(255, 255, 255);
	const sf::Color GAS_CELL_COLOR = sf::Color(0, 0, 0);
	const sf::Color WALL_CELL_COLOR = sf::Color(255, 92, 0);

	// Precomputed
	sf::VertexArray grid_vertices;
	sf::VertexArray grid_lines;
	sf::RectangleShape cell_shape;
	sf::RectangleShape grid_background;
	sf::RectangleShape ui_background;

	// Cells from previous frame
	uint16_t* previous_cells;	// Not currently used
	bool first_iteration;
	
public:
	/* Constructor */
	Visualization(int window_width, int window_height, int grid_width, int grid_height);
	~Visualization();

	/* Public Methods */
	void process_window_events();
	void init_grid();
	void manage_grid_update(Grid* grid, bool force_full_update = false);
	void update_grid_cell(Grid* grid, int cell_x, int cell_y);
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
	void update_whole_grid(Grid* grid);
	void update_grid_cells(Grid* grid);
	void find_grid_dimensions();
	void update_views();
	void handle_mouse_click(int mouse_x, int mouse_y);
	sf::Color adjust_gas_color(double concentration);
};