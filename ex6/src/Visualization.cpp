#include "Visualization.h"


/* Constructor */
Visualization::Visualization(int window_width, int window_height, int grid_width, int grid_height):
	window(sf::VideoMode(window_width, window_height), "LGA"),
	gui(window),
    cell_size(1),
    grid_width(grid_width),
    grid_height(grid_height)
{
    // Set grid view
    const float grid_view_width = (float)(window_width - UI_VIEW_WIDTH);
    const float grid_view_height = (float)window_height;

    grid_view.setViewport(sf::FloatRect(0.f, 0.f, grid_view_width / window_width, 1.f));
    grid_view.setSize(grid_view_width, grid_view_height);
    grid_view.setCenter(grid_view_width / 2, grid_view_height / 2);

    // Set ui view
    const float ui_view_width = (float)UI_VIEW_WIDTH;
    const float ui_view_height = (float)window_height;

    ui_view.setViewport(sf::FloatRect(ui_view_width / window_width, 0.f, ui_view_width / window_width, 1.f));
    ui_view.setSize(ui_view_width, ui_view_height);
    ui_view.setCenter(ui_view_width / 2, ui_view_height / 2);

    // Estimate cell size
    find_grid_dimensions();
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

        gui.handleEvent(event); // Process GUI events
    }
}

void Visualization::draw_grid()
{
    // Change rendering view to the grid
    window.setView(grid_view);

    // Draw grid background
    sf::RectangleShape background(sf::Vector2f(grid_view.getSize().x, grid_view.getSize().y));
    background.setFillColor(sf::Color(173, 216, 230)); // Light blue
    window.draw(background);

    // Draw grid cells
    float cell_internal_size = cell_size;
    float cell_outline_size = cell_size * CELL_OUTLINE_PORTION;

    sf::RectangleShape cell(sf::Vector2f(cell_internal_size, cell_internal_size));
    cell.setFillColor(sf::Color::Cyan);
    cell.setOutlineThickness(-cell_outline_size);
    cell.setOutlineColor(sf::Color::Black);

    for (int i = 0; i < grid_width; i++)
    {
        for (int j = 0; j < grid_height; j++)
        {
            float x = GRID_PADDING + i * cell_internal_size;
            float y = GRID_PADDING + j * cell_internal_size;

            cell.setPosition(x, y);
            window.draw(cell);
        }
    }
}

void Visualization::draw_ui()
{
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