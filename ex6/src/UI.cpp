#include "UI.h"
#include <iostream>

/* Constructor */
UI::UI(tgui::Gui& gui) : gui_ref(gui) {}

/* Public Methods */
void UI::initialize(float ui_offset_x, float ui_width) 
{
    // Layout configuration
    const float basic_margin = ui_width * 0.15;
    const float basic_width = ui_width - basic_margin * 2;
    const float basic_height = 60;
    const float basic_text_size = 40;
    const float small_text_size = 28;
    const float very_small_text_size = 22;
    const float top_margin = 20;

    const float half_button_width = (basic_width - basic_margin) / 2;
    const float total_x_offset = ui_offset_x + basic_margin;

    float y_index = 0;
    float current_y_pos = 0;
    float additional_vertical_gap_total = 0;

    // UI element configuration structure
    struct UIElementConfig 
    {
        std::string name;           // Element internal name
        std::string element_type;   // Type of element: "button", "label"
        std::string text;           // Text or label
        float text_size;            // Size of rendered text
        float width;                // Width of the element
        float x_offset;             // X-offset relative to total_x_offset
        bool advance_vertically;    // Whether this element starts a new row
        float y_offset;         // Additional vertical gap
    };

    // UI element definitions
    std::vector<UIElementConfig> element_configs = 
    {
        {"pause", "button", "Start", basic_text_size, basic_width, 0, true, 0},
        {"reset", "button", "Reset", basic_text_size, basic_width, 0, true, 0},
        {"slower", "button", "<", basic_text_size, half_button_width, 0,  true, 0},
        {"faster", "button", ">", basic_text_size, half_button_width, half_button_width + basic_margin, false, 0},
        {"speed_label", "label", "Speed: 5 UPS", small_text_size, ui_width, -basic_margin, true, 0},
        {"desc_label", "label", "Ctrl = 5 | Shift = 10", very_small_text_size, ui_width, -basic_margin, true, -40},
    };

    // Iterate through UI element configs
    for (const auto& config : element_configs) 
    {
        if (config.advance_vertically) 
        {
            y_index += 1;
            additional_vertical_gap_total += config.y_offset;
            current_y_pos = top_margin * y_index + basic_height * (y_index - 1) + additional_vertical_gap_total;
        }

        // Create UI elements based on type
        if (config.element_type == "button") 
        {
            auto button = tgui::Button::create(config.text);
            button->setSize(config.width, basic_height);
            button->setTextSize(config.text_size);
            button->setPosition(total_x_offset + config.x_offset, current_y_pos);
            buttons[config.name] = button;
            gui_ref.add(button);
        }
        else if (config.element_type == "label") 
        {
            auto label = tgui::Label::create(config.text);
            label->setSize(config.width, basic_height);
            label->setTextSize(config.text_size);
            label->setHorizontalAlignment(tgui::Label::HorizontalAlignment::Center);
            label->setPosition(total_x_offset + config.x_offset, current_y_pos);
            gui_ref.add(label);
        }
    }
}

/* Getters */
tgui::Button::Ptr UI::get_button(const std::string& name)
{
	return buttons[name];
}
