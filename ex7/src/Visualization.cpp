#include "Visualization.h"

/* Constructor */
Visualization::Visualization(int window_width, int window_height, int grid_width, int grid_height):
	main_window(sf::VideoMode(window_width, window_height), "LBM", sf::Style::Close),
	gui(main_window),
	main_grid_cell_size(1),
	grid_width(grid_width),
	grid_height(grid_height)
{
	/* Main Window */
	// Set main window attributes
	main_window.setFramerateLimit(0); // Disable the limit
	main_window.setVerticalSyncEnabled(false);

	// Set grid view
	const float grid_view_width = (float)(window_width * (1 - UI_VIEW_PORTION));
	const float grid_view_height = (float)window_height;

	grid_view.setViewport(sf::FloatRect(0.f, 0.f, grid_view_width / window_width, 1.f));
	grid_view.setSize(grid_view_width, grid_view_height);
	grid_view.setCenter(grid_view_width / 2, grid_view_height / 2);

	// Set ui view
	const float ui_view_width = (float)UI_VIEW_PORTION * window_width;
	const float ui_view_height = (float)window_height;

	ui_view.setViewport(sf::FloatRect(1 - UI_VIEW_PORTION, 0.f, UI_VIEW_PORTION, 1.f));
	ui_view.setSize(ui_view_width, ui_view_height);
	ui_view.setCenter(ui_view_width / 2, ui_view_height / 2);

	// Initialize visuals
	find_grid_dimensions();
	init_grid();
	init_ui();

	/* Sub Windows */
	
	// Create
	sub_window_vx.create(sf::VideoMode(grid_width * sub_grid_cell_size, grid_height * sub_grid_cell_size), "Velocity X-Axis", sf::Style::Close);
	sub_window_vy.create(sf::VideoMode(grid_width * sub_grid_cell_size, grid_height * sub_grid_cell_size), "Velocity Y-Axis", sf::Style::Close);

	// Set Attributes
	vx_window_visible = false;
	vy_window_visible = false;

	sf::View vx_view(sf::FloatRect(
		0.0f,
		0.0f,
		grid_width * sub_grid_cell_size,
		grid_height * sub_grid_cell_size
	));
	sub_window_vx.setView(vx_view);

	sf::View vy_view(sf::FloatRect(
		0.0f,
		0.0f,
		grid_width * sub_grid_cell_size,
		grid_height * sub_grid_cell_size
	));
	sub_window_vy.setView(vy_view);

	sub_window_vx.setVisible(vx_window_visible);
	sub_window_vy.setVisible(vy_window_visible);

	/* Misc */

	// Allocate memory for previous cells
	// This is literary pointless since the switch to LBM
	previous_cells = new uint16_t[grid_width * grid_height];
	first_iteration = true;
}

Visualization::~Visualization()
{
	delete[] previous_cells;
}

/* Public Methods */
void Visualization::process_window_events() 
{
	// Process events for the main window
	this->process_main_window();

	// Process vents for the sub windows
	this->process_sub_windows();
}

void Visualization::init_grid()
{
	/* Initialize The Vertex Array */

	// For the main grid
	main_grid_vertices.setPrimitiveType(sf::Quads);
	main_grid_vertices.resize(grid_width * grid_height * 4); // 4 vertices per cell

	// For the sub grids
	vx_grid_vertices.setPrimitiveType(sf::Quads);
	vx_grid_vertices.resize(grid_width * grid_height * 4); // 4 vertices per cell    
	
	vy_grid_vertices.setPrimitiveType(sf::Quads);
	vy_grid_vertices.resize(grid_width * grid_height * 4); // 4 vertices per cell

	/* Compute Corners For Each Cell */
	for (int i = 0; i < grid_width; i++) 
	{
		for (int j = 0; j < grid_height; j++) 
		{
			int cell_id = j * grid_width + i;

			/* Main Grid */

			// Compute four corners
			float x = GRID_PADDING + i * main_grid_cell_size;
			float y = GRID_PADDING + j * main_grid_cell_size;

			sf::Vertex* main_grid_quad = &main_grid_vertices[cell_id * 4];

			main_grid_quad[0].position = sf::Vector2f(x, y);
			main_grid_quad[1].position = sf::Vector2f(x + main_grid_cell_size, y);
			main_grid_quad[2].position = sf::Vector2f(x + main_grid_cell_size, y + main_grid_cell_size);
			main_grid_quad[3].position = sf::Vector2f(x, y + main_grid_cell_size);

			// Set initial color
			sf::Color initial_color = EMPTY_CELL_COLOR;
			main_grid_quad[0].color = initial_color;
			main_grid_quad[1].color = initial_color;
			main_grid_quad[2].color = initial_color;
			main_grid_quad[3].color = initial_color;

			/* Sub Grids - vx */

			// Compute four corners
			x = i * sub_grid_cell_size;
			y = j * sub_grid_cell_size;

			sf::Vertex* vx_grid_quad = &vx_grid_vertices[cell_id * 4];

			vx_grid_quad[0].position = sf::Vector2f(x, y);
			vx_grid_quad[1].position = sf::Vector2f(x + sub_grid_cell_size, y);
			vx_grid_quad[2].position = sf::Vector2f(x + sub_grid_cell_size, y + sub_grid_cell_size);
			vx_grid_quad[3].position = sf::Vector2f(x, y + sub_grid_cell_size);

			// Set initial color
			initial_color = sf::Color::Red;
			vx_grid_quad[0].color = initial_color;
			vx_grid_quad[1].color = initial_color;
			vx_grid_quad[2].color = initial_color;
			vx_grid_quad[3].color = initial_color;            
			
			/* Sub Grids - vy */

			// Compute four corners
			x = i * sub_grid_cell_size;
			y = j * sub_grid_cell_size;

			sf::Vertex* vy_grid_quad = &vy_grid_vertices[cell_id * 4];

			vy_grid_quad[0].position = sf::Vector2f(x, y);
			vy_grid_quad[1].position = sf::Vector2f(x + sub_grid_cell_size, y);
			vy_grid_quad[2].position = sf::Vector2f(x + sub_grid_cell_size, y + sub_grid_cell_size);
			vy_grid_quad[3].position = sf::Vector2f(x, y + sub_grid_cell_size);

			// Set initial color
			initial_color = sf::Color::Green;
			vy_grid_quad[0].color = initial_color;
			vy_grid_quad[1].color = initial_color;
			vy_grid_quad[2].color = initial_color;
			vy_grid_quad[3].color = initial_color;
		}
	}

	// Initialize the vertex array for the grid lines
	grid_lines.setPrimitiveType(sf::Lines);
	grid_lines.resize((grid_width + 1) * 2 + (grid_height + 1) * 2); // Horizontal + Vertical lines

	int idx = 0;

	// Horizontal lines
	for (int j = 0; j <= grid_height; j++) 
	{
		float y = GRID_PADDING + j * main_grid_cell_size;

		grid_lines[idx].position = sf::Vector2f(GRID_PADDING, y);
		grid_lines[idx].color = sf::Color::Black;
		idx++;

		grid_lines[idx].position = sf::Vector2f(GRID_PADDING + grid_width * main_grid_cell_size, y);
		grid_lines[idx].color = sf::Color::Black;
		idx++;
	}

	// Vertical lines
	for (int i = 0; i <= grid_width; i++) 
	{
		float x = GRID_PADDING + i * main_grid_cell_size;

		grid_lines[idx].position = sf::Vector2f(x, GRID_PADDING);
		grid_lines[idx].color = sf::Color::Black;
		idx++;

		grid_lines[idx].position = sf::Vector2f(x, GRID_PADDING + grid_height * main_grid_cell_size);
		grid_lines[idx].color = sf::Color::Black;
		idx++;
	}

	// Initialize grid background
	grid_background.setSize(sf::Vector2f(grid_view.getSize().x, grid_view.getSize().y));
	grid_background.setFillColor(sf::Color(173, 216, 230)); // Light blue
}

void Visualization::manage_grid_update(Grid* grid, bool force_full_update)
{
	force_full_update = true;   // Overwrite for debugging
	if (first_iteration || force_full_update)
		update_whole_grid(grid);
	else
		update_grid_cells(grid);
}

void Visualization::update_whole_grid(Grid* grid)
{
	// Update the color of each cell in the vertex array
	for (int i = 0; i < grid_width; i++)
	{
		for (int j = 0; j < grid_height; j++)
		{
			update_grid_cell(grid, i, j);
		}
	}

	first_iteration = false;
}

void Visualization::update_grid_cells(Grid* grid)
{
	for (int i = 0; i < grid_width; i++)
	{
		for (int j = 0; j < grid_height; j++)
		{
			update_grid_cell(grid, i, j);
		}
	}
}

void Visualization::compute_and_draw_stream_lines(Grid* grid, int spacing, float scale)
{
	// Iterate through the grid with given spacing
	for (int x = spacing; x < grid_width; x += spacing)
	{
		for (int y = spacing; y < grid_height; y += spacing)
		{
			/* Calculate start and end of the line */
			// Get velocity
			int cell_id = grid->get_id(x, y);
			float vx = grid->get_velocity_x(cell_id);
			float vy = grid->get_velocity_y(cell_id);

			// Compute the scaled end position of the line
			float start_x = GRID_PADDING + x * this->main_grid_cell_size;
			float start_y = GRID_PADDING + y * this->main_grid_cell_size;

			float end_x = start_x + scale * vx;
			float end_y = start_y + scale * vy;

			/* Ensure the line is within boundries of the grid */
			if (start_x < GRID_PADDING || start_x > GRID_PADDING + grid_width * this->main_grid_cell_size ||
				start_y < GRID_PADDING || start_y > GRID_PADDING + grid_height * this->main_grid_cell_size)
			{
				continue; // Skip drawing this line if the start is outside the grid
			}

			// Adjust end points if they are outside the grid
			if (end_x < GRID_PADDING)
			{
				end_x = GRID_PADDING;
			}
			else if (end_x > GRID_PADDING + grid_width * this->main_grid_cell_size)
			{
				end_x = GRID_PADDING + grid_width * this->main_grid_cell_size;
			}

			if (end_y < GRID_PADDING)
			{
				end_y = GRID_PADDING;
			}
			else if (end_y > GRID_PADDING + grid_height * this->main_grid_cell_size)
			{
				end_y = GRID_PADDING + grid_height * this->main_grid_cell_size;
			}

			/* Draw The Line */
			// Use rectangle shape for thcness
			sf::RectangleShape line;
			float length = sqrt((end_x - start_x) * (end_x - start_x) + (end_y - start_y) * (end_y - start_y));
			line.setSize(sf::Vector2f(length, this->STREAMLINE_THICKNESS));
			line.setFillColor(this->STREAMLINE_COLOR);

			// Rotate the line to align with the velocity direction
			float angle = atan2(end_y - start_y, end_x - start_x) * 180.f / 3.14f;
			line.setRotation(angle);

			// Position the line at the start point
			line.setPosition(sf::Vector2f(start_x, start_y));

			// Draw the line
			main_window.draw(line);
		}
	}
}

void Visualization::update_grid_cell(Grid* grid, int cell_x, int cell_y)
{
	// Find the id
	int cell_id = grid->get_id(cell_x, cell_y);

	// Call the updater
	update_grid_cell(grid, cell_id);
}

void Visualization::update_grid_cell(Grid* grid, int cell_id)
{
	/* Main Grid */

	// Decide on the colour
	sf::Color cell_color = EMPTY_CELL_COLOR;

	if (cell_id == followed_cell)
		cell_color = FOLLOWED_CELL_COLOR;

	else if (grid->get_cell_is_wall(cell_id))
		cell_color = WALL_CELL_COLOR;

	else if (grid->get_cell_concetration(cell_id) > 1e-6)
		cell_color = get_gas_color(grid->get_cell_concetration(cell_id));

	// Apply the colour
	sf::Vertex* quad = &main_grid_vertices[cell_id * 4];
	quad[0].color = cell_color;
	quad[1].color = cell_color;
	quad[2].color = cell_color;
	quad[3].color = cell_color;

	/* Sub Grid - vx */

	// Decide on the color
	cell_color = get_velocity_color(grid->get_velocity_x(cell_id));

	// Apply the colour
	quad = &vx_grid_vertices[cell_id * 4];
	quad[0].color = cell_color;
	quad[1].color = cell_color;
	quad[2].color = cell_color;
	quad[3].color = cell_color;

	/* Sub Grid - vy */

	// Decide on the color
	cell_color = get_velocity_color(grid->get_velocity_y(cell_id));

	// Apply the colour
	quad = &vy_grid_vertices[cell_id * 4];
	quad[0].color = cell_color;
	quad[1].color = cell_color;
	quad[2].color = cell_color;
	quad[3].color = cell_color;

}

void Visualization::draw_grid(Grid* grid, bool draw_grid_lines, bool draw_stream_lines)
{
	// Change rendering view to the grid
	main_window.setView(grid_view);

	// Draw grid background
	main_window.draw(grid_background);

	// Draw the grid cells
	main_window.draw(main_grid_vertices);

	// Draw grid lines
	if(draw_grid_lines)
		main_window.draw(grid_lines);

	// Draw streamlines
	if (draw_stream_lines)
		compute_and_draw_stream_lines(grid, STREAMLINE_SPACING, STREAMLINE_SCALE);
}

void Visualization::draw_particles(Grid::Particle* particles, int num)
{
	sf::CircleShape circle;

	for (int i = 0; i < num; i++)
	{
		/* Draw Trajectory */
		if (particles[i].trajectory.size() > 1)
		{
			// Create a line from points
			sf::VertexArray line(sf::LinesStrip, particles[i].trajectory.size());
			for (int j = 0; j < particles[i].trajectory.size(); j += 1)
			{
				line[j].position = sf::Vector2f(
					particles[i].trajectory[j].x + GRID_PADDING,
					particles[i].trajectory[j].y + GRID_PADDING
				);

				// Use the particle's color
				line[j].color = particles[i].color;
			}

			// Draw the line
			main_window.draw(line);
		}

		// Set properties
		circle.setRadius(3.f);
		circle.setFillColor(particles[i].color);
		circle.setPosition(particles[i].x + GRID_PADDING, particles[i].y + GRID_PADDING);

		// Draw
		main_window.draw(circle);
	}
}

void Visualization::init_ui()
{
	// Initialize UI background
	ui_background.setSize(sf::Vector2f(ui_view.getSize().x, ui_view.getSize().y));
	ui_background.setFillColor(sf::Color::Green);
}

void Visualization::draw_ui()
{
	// Change the view to UI section
	main_window.setView(ui_view);

	// Draw UI background
	main_window.draw(ui_background);

	// Update the GUI
	gui.draw();
}

void Visualization::draw_sub_windows()
{
	if (vx_window_visible)
	{
		sub_window_vx.draw(vx_grid_vertices);
	}
	
	if (vy_window_visible)
	{
		sub_window_vy.draw(vy_grid_vertices);
	}

}

void Visualization::clear()
{
	// Clear the main window
	main_window.clear(sf::Color::Black);

	// Clear the sub windows
	if (vx_window_visible)
		sub_window_vx.clear(sf::Color::White);

	if (vy_window_visible)
		sub_window_vy.clear(sf::Color::White);
}

void Visualization::display()
{
	// Display the main window
	main_window.display();

	// Display the sub windows
	if (vx_window_visible)
		sub_window_vx.display();

	if (vy_window_visible)
		sub_window_vy.display();
}

/* Getters */
bool Visualization::is_window_open() const
{
	return main_window.isOpen();
}

float Visualization::get_ui_view_offset() const
{
	return main_window.getSize().x * ui_view.getViewport().left;
}

float Visualization::get_ui_view_width() const
{
	return main_window.getSize().x * ui_view.getViewport().width;
}

bool Visualization::is_vx_visible()
{
	return this->vx_window_visible;
}

bool Visualization::is_vy_visible()
{
	return this->vy_window_visible;
}

tgui::Gui& Visualization::get_gui()
{
	return gui;
}

double Visualization::get_cell_size()
{
	return main_grid_cell_size;
}

/* Setters */

// Callbacks
void Visualization::set_cell_modify_callback(std::function<void(int, int)> callback)
{
	this->cell_modify_callback = callback;
}

void Visualization::set_cell_follow_callback(std::function<void(int, int)> callback)
{
	this->cell_follow_callback = callback;
}

void Visualization::set_update_buttons_callback(std::function<void()> callback)
{
	this->update_buttons_callback = callback;
}

void Visualization::set_vx_window_visibility(bool value)
{
	vx_window_visible = value;
	sub_window_vx.setVisible(vx_window_visible);
}

void Visualization::set_vy_window_visibility(bool value)
{
	vy_window_visible = value;
	sub_window_vy.setVisible(vy_window_visible);
}

void Visualization::set_followed_cell(int id)
{
	this->followed_cell = id;
}

/* Private Methods */
void Visualization::process_main_window()
{
	sf::Event event;
	while (main_window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			main_window.close();
		}
		if (event.type == sf::Event::Resized)
		{
			update_views();
		}
		if (event.type == sf::Event::MouseButtonPressed)
		{
			if (event.mouseButton.button == sf::Mouse::Left)
			{
				handle_mouse_click(event.mouseButton.x, event.mouseButton.y, true);
			}
			if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
			{
				handle_mouse_click(event.mouseButton.x, event.mouseButton.y, false);
			}
		}
		if (event.type == sf::Event::MouseMoved)
		{
			if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
			{
				handle_mouse_click(event.mouseMove.x, event.mouseMove.y, true);
			}
		}

		gui.handleEvent(event); // Process GUI events
	}
}

void Visualization::process_sub_windows()
{
	sf::Event event;

	/* X-Axis Sub Window */
	if (vx_window_visible)
	{
		while (sub_window_vx.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				this->set_vx_window_visibility(false);
				this->update_buttons_callback();
			}
		}
	}    
	
	/* Y-Axis Sub Window */
	if (vy_window_visible)
	{
		while (sub_window_vy.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				this->set_vy_window_visibility(false);
				this->update_buttons_callback();
			}
		}
	}
}

// Estimate cell size based on window size, padding and number of cells
void Visualization::find_grid_dimensions()
{
	sf::Vector2u window_size = main_window.getSize();

	// Find total space for the grid without padding
	float available_width = grid_view.getSize().x - 2 * GRID_PADDING;
	float available_height = grid_view.getSize().y - 2 * GRID_PADDING;

	// Minimal size of a cell
	main_grid_cell_size = std::min(available_width / grid_width, available_height / grid_height);
	sub_grid_cell_size = main_grid_cell_size * 0.5f;
}

void Visualization::update_views()
{
	find_grid_dimensions();
}

void Visualization::handle_mouse_click(int mouse_x, int mouse_y, bool left_button)
{
	// Convert window coords into grid coords
	sf::Vector2f world_coords = main_window.mapPixelToCoords(sf::Vector2i(mouse_x, mouse_y), grid_view);

	// Find the index of the cell
	int cell_x = (world_coords.x - GRID_PADDING) / main_grid_cell_size;
	int cell_y = (world_coords.y - GRID_PADDING) / main_grid_cell_size;

	// Check if the click happened inside the grid
	if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height)
	{
		if (left_button)
			cell_modify_callback(cell_x, cell_y);
		else
			cell_follow_callback(cell_x, cell_y);
	}
}

void Visualization::save_grid_as_image(std::string path, int iteration)
{
	// Create render textures
	sf::RenderTexture main_grid_tex;
	sf::RenderTexture vx_tex;
	sf::RenderTexture vy_tex;

	main_grid_tex.create(grid_width * main_grid_cell_size, grid_height * main_grid_cell_size);
	vx_tex.create(grid_width * sub_grid_cell_size, grid_height * sub_grid_cell_size);
	vy_tex.create(grid_width * sub_grid_cell_size, grid_height * sub_grid_cell_size);

	// Clear render texutures
	main_grid_tex.clear();
	vx_tex.clear();
	vy_tex.clear();

	// Draw vertices onto the texture
	
	// Copy main grid verts with offset
	sf::VertexArray main_grid_copy(sf::Quads);
	int vertices = grid_width * grid_height * 4;
	main_grid_copy.resize(vertices);

	for (int i = 0; i < vertices; i++)
	{
		// Copy vert
		main_grid_copy[i] = main_grid_vertices[i];

		// Offset copy
		main_grid_copy[i].position.x -= GRID_PADDING;
		main_grid_copy[i].position.y -= GRID_PADDING;
	}

	main_grid_tex.draw(main_grid_copy);
	vx_tex.draw(vx_grid_vertices);
	vy_tex.draw(vy_grid_vertices);

	// Display vertices on the texture
	main_grid_tex.display();
	vx_tex.display();
	vy_tex.display();

	// Create images based on textures
	sf::Image main_grid_img = main_grid_tex.getTexture().copyToImage();
	sf::Image vx_img = vx_tex.getTexture().copyToImage();
	sf::Image vy_img = vy_tex.getTexture().copyToImage();

	// Save images to files
	std::string filename = path + "i=" + std::to_string(iteration) + "-";
	std::string extension = ".png";

	bool main_grid_success = main_grid_img.saveToFile(filename + "a_main_grid" + extension);
	bool vx_success = vx_img.saveToFile(filename + "b_vx" + extension);
	bool vy_success = vy_img.saveToFile(filename + "c_vy" + extension);

	if (main_grid_success && vx_success && vy_success)
	{
		std::cout << "Saved as set of images at: " + filename << "\n";
	}

	// Else not needed saveToFile prints to console
}

sf::Color Visualization::get_gas_color(double concentration)
{
	// Normalize concentration (0 to 1)
	// double normalized_conc = std::min(1.0, concentration / (double)Grid::direction_num);
	double normalized_conc = std::min(1.0, concentration);

	// Linear interpolation for each color channel
	sf::Color cell_color = EMPTY_CELL_COLOR;
	cell_color.r = static_cast<sf::Uint8>(EMPTY_CELL_COLOR.r + normalized_conc * (GAS_CELL_COLOR.r - EMPTY_CELL_COLOR.r));
	cell_color.g = static_cast<sf::Uint8>(EMPTY_CELL_COLOR.g + normalized_conc * (GAS_CELL_COLOR.g - EMPTY_CELL_COLOR.g));
	cell_color.b = static_cast<sf::Uint8>(EMPTY_CELL_COLOR.b + normalized_conc * (GAS_CELL_COLOR.b - EMPTY_CELL_COLOR.b));

	return cell_color;
}

sf::Color Visualization::get_velocity_color(double velocity)
{
	// Special coloring for areas with no velocity
	if (abs(velocity) < NO_VELOCITY_BOUNDARY)
	{
		return NO_VELOCITY_COLOR;
	}

	// Color selection
	sf::Color color = POSITIVE_VELOCITY_COLOR;

	if (velocity < 0)
	{
		// Pick the color
		color = NEGATIVE_VELOCITY_COLOR;

		// Apply absolute value
		velocity = -velocity;
	}

	// Velocity should always be between <0, 1> at this point
	// But it will rarely reach the max of 1
	velocity = clamp(velocity / VELOCITY_MAX, 0.f, 1.f);

	// Interpolate between NO_VELOCITY_COLOR and the target color
	sf::Color interpolated_color;
	interpolated_color.r = static_cast<sf::Uint8>(NO_VELOCITY_COLOR.r + velocity * (color.r - NO_VELOCITY_COLOR.r));
	interpolated_color.g = static_cast<sf::Uint8>(NO_VELOCITY_COLOR.g + velocity * (color.g - NO_VELOCITY_COLOR.g));
	interpolated_color.b = static_cast<sf::Uint8>(NO_VELOCITY_COLOR.b + velocity * (color.b - NO_VELOCITY_COLOR.b));

	return interpolated_color;
}

double Visualization::clamp(double value, double min, double max)
{
	if (value > max)
		return max;
	else if (value < min)
		return min;

	return value;
}
