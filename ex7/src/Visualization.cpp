#include "Visualization.h"

/* Constructor */
Visualization::Visualization(int window_width, int window_height, int grid_width, int grid_height):
	window(sf::VideoMode(window_width, window_height), "LBM", sf::Style::Close),
	gui(window),
    cell_size(1),
    grid_width(grid_width),
    grid_height(grid_height)
{
    // Set window attributes
    window.setFramerateLimit(0); // Limit to 120 FPS
    window.setVerticalSyncEnabled(false);

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

    // Allocate memory for previous cells
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
    while (window.pollEvent(event)) 
    {
        if (event.type == sf::Event::Closed) 
        {
            window.close();
        }
        if (event.type == sf::Event::Resized)
        {
            update_views();
        }
        if (event.type == sf::Event::MouseButtonPressed) 
        {
            if (event.mouseButton.button == sf::Mouse::Left) 
            {
                handle_mouse_click(event.mouseButton.x, event.mouseButton.y);
            }
        }
        if (event.type == sf::Event::MouseMoved) 
        {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) 
            {
                handle_mouse_click(event.mouseMove.x, event.mouseMove.y);
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

            // Initial color (default, can be updated later)
            // sf::Color initial_color = state_to_color(Automaton::EMPTY);
            // quad[0].color = initial_color;
            // quad[1].color = initial_color;
            // quad[2].color = initial_color;
            // quad[3].color = initial_color;
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

void Visualization::manage_grid_update(Automaton::Grid* grid, bool force_full_update)
{
    if (first_iteration || force_full_update)
        update_whole_grid(grid);
    else
        update_grid_cells(grid);
}

void Visualization::update_whole_grid(Automaton::Grid* grid)
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

void Visualization::update_grid_cells(Automaton::Grid* grid)
{
    for (int i = 0; i < grid_width; i++)
    {
        for (int j = 0; j < grid_height; j++)
        {
            update_grid_cell(grid, i, j);
        }
    }
}

void Visualization::update_grid_cell(Automaton::Grid* grid, int cell_x, int cell_y)
{
    // Calculate cell ID in row-major order
    int cell_id = cell_y * grid_width + cell_x;

    // Check if the cell requires updating
    // if
    //  return

    // Decide on the colour
    sf::Color cell_color = EMPTY_CELL_COLOR;

    if (grid->is_wall[cell_id])
        cell_color = WALL_CELL_COLOR;

    else if (grid->concentration[cell_id] > 1e-6)
        cell_color = GAS_CELL_COLOR;

    // Apply the colour
    sf::Vertex* quad = &grid_vertices[cell_id * 4];
    quad[0].color = cell_color;
    quad[1].color = cell_color;
    quad[2].color = cell_color;
    quad[3].color = cell_color;

    // Mark the cell as recently updated
}

void Visualization::draw_grid(bool draw_grid_lines)
{
    // Change rendering view to the grid
    window.setView(grid_view);

    // Draw grid background
    window.draw(grid_background);

    // Draw the grid cells
    window.draw(grid_vertices);

    // Draw grid lines
    if(draw_grid_lines)
        window.draw(grid_lines);
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
    window.setView(ui_view);

    // Draw UI background
    window.draw(ui_background);

    // Update the GUI
    gui.draw();
}

void Visualization::clear()
{
    window.clear(sf::Color::Black);
}

void Visualization::display()
{
    window.display();
}

/* Getters */
bool Visualization::is_window_open() const
{
    return window.isOpen();
}

float Visualization::get_ui_view_offset() const
{
    return window.getSize().x * ui_view.getViewport().left;
}

float Visualization::get_ui_view_width() const
{
    return window.getSize().x * ui_view.getViewport().width;
}

tgui::Gui& Visualization::get_gui()
{
    return gui;
}

/* Setters */
void Visualization::set_cell_click_callback(std::function<void(int, int)> callback)
{
    this->cell_click_callback = callback;
}

/* Private Methods */
// Estimate cell size based on window size, padding and number of cells
void Visualization::find_grid_dimensions()
{
    sf::Vector2u window_size = window.getSize();

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

void Visualization::handle_mouse_click(int mouse_x, int mouse_y)
{
    // Convert window coords into grid coords
    sf::Vector2f world_coords = window.mapPixelToCoords(sf::Vector2i(mouse_x, mouse_y), grid_view);

    // Find the index of the cell
    int cell_x = (world_coords.x - GRID_PADDING) / cell_size;
    int cell_y = (world_coords.y - GRID_PADDING) / cell_size;

    // Check if the click happened inside the grid
    if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height)
        cell_click_callback(cell_x, cell_y);
}
