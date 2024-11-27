#include "Visualization.h"

/* Constructor */
Visualization::Visualization(int window_width, int window_height, int grid_width, int grid_height):
	window(sf::VideoMode(window_width, window_height), "LGA", sf::Style::Close),
	gui(window),
    cell_size(1),
    grid_width(grid_width),
    grid_height(grid_height)
{
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

void Visualization::draw_grid()
{
    // Change rendering view to the grid
    window.setView(grid_view);

    // Draw grid background
    sf::RectangleShape background(sf::Vector2f(grid_view.getSize().x, grid_view.getSize().y));
    background.setFillColor(sf::Color(173, 216, 230)); // Light blue
    window.draw(background);

    // Draw grid cells
    float cell_outline_size = cell_size * CELL_OUTLINE_PORTION;

    sf::RectangleShape cell(sf::Vector2f(cell_size, cell_size));
    cell.setFillColor(sf::Color::Cyan);
    cell.setOutlineThickness(-cell_outline_size);
    cell.setOutlineColor(sf::Color::Black);

    for (int i = 0; i < grid_width; i++)
    {
        for (int j = 0; j < grid_height; j++)
        {
            float x = GRID_PADDING + i * cell_size;
            float y = GRID_PADDING + j * cell_size;

            cell.setPosition(x, y);
            window.draw(cell);
        }
    }
}

void Visualization::draw_ui()
{
    // Change the view to UI section
    window.setView(ui_view);

    // Draw background
    sf::RectangleShape background(sf::Vector2f(ui_view.getSize().x, ui_view.getSize().y));
    background.setFillColor(sf::Color::Green);
    window.draw(background);

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
        std::cout << "Clicked cell: (" << cell_x << ", " << cell_y << ")\n";
}
