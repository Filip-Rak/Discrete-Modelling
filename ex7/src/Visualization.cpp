#include "Visualization.h"

/* Constructor */
Visualization::Visualization(int window_width, int window_height, int grid_width, int grid_height):
	main_window(sf::VideoMode(window_width, window_height), "LBM", sf::Style::Close),
    sub_window_vx(sf::VideoMode(1, 1), "Velocity X-Axis", sf::Style::Close),
    sub_window_vy(sf::VideoMode(1, 1), "Velocity Y-Axis", sf::Style::Close),
	gui(main_window),
    cell_size(1),
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
    // Set Attributes
    vx_window_visible = false;
    vy_window_visible = false;

    sub_window_vx.setVisible(vx_window_visible);
    sub_window_vy.setVisible(vy_window_visible);

    sf::Vector2u size(grid_width * cell_size, grid_height * cell_size);
    sub_window_vx.setSize(size);
    sub_window_vy.setSize(size);

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

void Visualization::init_grid()
{
    // Initialize the vertex array for the grid
    grid_vertices.setPrimitiveType(sf::Quads);
    grid_vertices.resize(grid_width * grid_height * 4); // 4 vertices per cell

    for (int i = 0; i < grid_width; i++) {
        for (int j = 0; j < grid_height; j++) {
            int cell_id = j * grid_width + i;

            // Compute the four corners of the cell
            float x = GRID_PADDING + i * cell_size;
            float y = GRID_PADDING + j * cell_size;

            sf::Vertex* quad = &grid_vertices[cell_id * 4];

            quad[0].position = sf::Vector2f(x, y);
            quad[1].position = sf::Vector2f(x + cell_size, y);
            quad[2].position = sf::Vector2f(x + cell_size, y + cell_size);
            quad[3].position = sf::Vector2f(x, y + cell_size);

            // Initial color
            sf::Color initial_color = EMPTY_CELL_COLOR;
            quad[0].color = initial_color;
            quad[1].color = initial_color;
            quad[2].color = initial_color;
            quad[3].color = initial_color;
        }
    }

    // Initialize the vertex array for the grid lines
    grid_lines.setPrimitiveType(sf::Lines);
    grid_lines.resize((grid_width + 1) * 2 + (grid_height + 1) * 2); // Horizontal + Vertical lines

    int idx = 0;

    // Horizontal lines
    for (int j = 0; j <= grid_height; j++) {
        float y = GRID_PADDING + j * cell_size;

        grid_lines[idx].position = sf::Vector2f(GRID_PADDING, y);
        grid_lines[idx].color = sf::Color::Black;
        idx++;

        grid_lines[idx].position = sf::Vector2f(GRID_PADDING + grid_width * cell_size, y);
        grid_lines[idx].color = sf::Color::Black;
        idx++;
    }

    // Vertical lines
    for (int i = 0; i <= grid_width; i++) {
        float x = GRID_PADDING + i * cell_size;

        grid_lines[idx].position = sf::Vector2f(x, GRID_PADDING);
        grid_lines[idx].color = sf::Color::Black;
        idx++;

        grid_lines[idx].position = sf::Vector2f(x, GRID_PADDING + grid_height * cell_size);
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

void Visualization::update_grid_cell(Grid* grid, int cell_x, int cell_y)
{
    // Check if the cell requires updating
    // if
    //  return

    // Find the id
    int cell_id = grid->get_id(cell_x, cell_y);

    // Decide on the colour
    sf::Color cell_color = EMPTY_CELL_COLOR;

    if (grid->get_cell_is_wall(cell_id))
        cell_color = WALL_CELL_COLOR;

    else if (grid->get_cell_concetration(cell_id) > 1e-6)
        cell_color = adjust_gas_color(grid->get_cell_concetration(cell_id));

    // Apply the colour
    sf::Vertex* quad = &grid_vertices[cell_id * 4];
    quad[0].color = cell_color;
    quad[1].color = cell_color;
    quad[2].color = cell_color;
    quad[3].color = cell_color;

    // Mark the cell as recently updated
    // (...)
}

void Visualization::draw_grid(bool draw_grid_lines)
{
    // Change rendering view to the grid
    main_window.setView(grid_view);

    // Draw grid background
    main_window.draw(grid_background);

    // Draw the grid cells
    main_window.draw(grid_vertices);

    // Draw grid lines
    if(draw_grid_lines)
        main_window.draw(grid_lines);
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
    // Nothing to draw yet
    // sub_window_vx.draw();
    // sub_window_vy.draw();
}

void Visualization::clear()
{
    // Clear the main window
    main_window.clear(sf::Color::Black);

    // Clear the sub windows
    if (vx_window_visible)
        sub_window_vx.clear(sf::Color::Black);

    if (vy_window_visible)
        sub_window_vy.clear(sf::Color::Blue);
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

tgui::Gui& Visualization::get_gui()
{
    return gui;
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

/* Private Methods */
// Estimate cell size based on window size, padding and number of cells
void Visualization::find_grid_dimensions()
{
    sf::Vector2u window_size = main_window.getSize();

    // Find total space for the grid without padding
    float available_width = grid_view.getSize().x - 2 * GRID_PADDING;
    float available_height = grid_view.getSize().y - 2 * GRID_PADDING;

    // Minimal size of a cell
    cell_size = std::min(available_width / grid_width, available_height / grid_height);
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
    int cell_x = (world_coords.x - GRID_PADDING) / cell_size;
    int cell_y = (world_coords.y - GRID_PADDING) / cell_size;

    // Check if the click happened inside the grid
    if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height)
    {
        if (left_button)
            cell_modify_callback(cell_x, cell_y);
        else
            cell_follow_callback(cell_x, cell_y);
    }
}

sf::Color Visualization::adjust_gas_color(double concentration)
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
