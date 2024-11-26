#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/TGUI.hpp>

#include "kernel-test.h"

#include <iostream>

int main_test() {
    // Create an SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML + TGUI Test");

    // Attach a TGUI GUI to the SFML window
    tgui::Gui gui(window);

    // Create a TGUI button
    auto button = tgui::Button::create("Click Me");
    button->setPosition(350, 250); // Set button position
    button->setSize(100, 50);      // Set button size
    button->onClick([]() {
        // std::cout << "Button clicked!" << std::endl;
        run();
        });

    // Add the button to the GUI
    gui.add(button);

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            // Close the window when requested
            if (event.type == sf::Event::Closed)
                window.close();

            // Pass events to TGUI
            gui.handleEvent(event);
        }

        // Clear the window
        window.clear(sf::Color::Black);

        // Draw the TGUI GUI
        gui.draw();

        // Display the window contents
        window.display();
    }

    return 0;
}
