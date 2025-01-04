#pragma once

#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include "Grid.h"

class Visualization
{
public:
	/* Enums */
	enum CellVisualState
	{
		EMPTY,
		FLUID, 
		WALL
	};

private:
	/* Attributes */

	// Components
	sf::RenderWindow main_window;
	sf::RenderWindow sub_window_vx;
	sf::RenderWindow sub_window_vy;
	tgui::Gui gui;

	// Views
	sf::View grid_view;
	sf::View ui_view;

	// Flags
	bool vx_window_visible;
	bool vy_window_visible;
	int followed_cell = -1;

	// Constants
	const float UI_VIEW_PORTION = 0.25f;
	const float GRID_PADDING = 20.f;
	const float VELOCITY_MAX = 0.02f;	// 0.02f for BC1 
	const float NO_VELOCITY_BOUNDARY = 1e-8f;	// Use higher 1e-6 for V1 BCs
	const float STREAMLINE_THICKNESS = 1.5f;
	const float STREAMLINE_SPACING = 10.f;
	const float STREAMLINE_SCALE = 100.f;

	// Settings
	float main_grid_cell_size;
	float sub_grid_cell_size;
	int grid_width;
	int grid_height;
	std::function<void(int, int)> cell_modify_callback;
	std::function<void(int, int)> cell_follow_callback;
	std::function<void()> update_buttons_callback;

	// Colours
	const sf::Color EMPTY_CELL_COLOR = sf::Color(255, 255, 255);
	const sf::Color GAS_CELL_COLOR = sf::Color(0, 0, 255);
	const sf::Color WALL_CELL_COLOR = sf::Color(255, 92, 0);
	const sf::Color FOLLOWED_CELL_COLOR = sf::Color(255, 0, 0);
	const sf::Color POSITIVE_VELOCITY_COLOR = sf::Color(255, 0, 0);
	const sf::Color NEGATIVE_VELOCITY_COLOR = sf::Color(0, 0, 255);
	const sf::Color NO_VELOCITY_COLOR = sf::Color(255, 255, 255);
	const sf::Color STREAMLINE_COLOR = sf::Color(0, 0, 0);

	// Precomputed
	sf::VertexArray main_grid_vertices;
	sf::VertexArray vx_grid_vertices;
	sf::VertexArray vy_grid_vertices;
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
	void update_grid_cell(Grid* grid, int cell_id);
	void draw_grid(Grid* grid, bool draw_grid_lines, bool draw_streamlines);
	void draw_particles(Grid::Particle* particles, int num);
	void draw_ui();
	void draw_sub_windows();
	void init_ui();
	void clear();
	void display();
	void save_grid_as_image(std::string path, int iteration);

	/* Getters */
	bool is_window_open() const;
	float get_ui_view_offset() const;
	float get_ui_view_width() const;
	bool is_vx_visible();
	bool is_vy_visible();
	tgui::Gui& get_gui();
	double get_cell_size();

	/* Setters */

	// Callbacks
	void set_cell_modify_callback(std::function<void(int, int)> callback);
	void set_cell_follow_callback(std::function<void(int, int)> callback);
	void set_update_buttons_callback(std::function<void()> callback);

	void set_vx_window_visibility(bool value);
	void set_vy_window_visibility(bool value);
	void set_followed_cell(int id);

private:
	/* Private Methods */
	void process_main_window();
	void process_sub_windows();
	void update_whole_grid(Grid* grid);
	void update_grid_cells(Grid* grid);
	void compute_and_draw_stream_lines(Grid* grid, int spacing, float scale);
	void find_grid_dimensions();
	void update_views();
	void handle_mouse_click(int mouse_x, int mouse_y, bool left_mouse);
	sf::Color get_gas_color(double concentration);
	sf::Color get_velocity_color(double velocity);
	double clamp(double value, double min, double max);
};