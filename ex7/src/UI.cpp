#include "UI.h"
#include <iostream>

/* Constructor */
UI::UI(tgui::Gui& gui) : gui_ref(gui) {}

/* Public Methods */
void UI::initialize(float ui_offset_x, float ui_width, float ctrl_speed, float shift_speed)
{
    // Layout configuration
    const float basic_margin = ui_width * 0.10;
    const float basic_width = ui_width - basic_margin * 2;
    const float half_button_gap = basic_margin * 0.5;
    const float half_button_width = basic_width / 2.f - half_button_gap * 0.5f;
    const float basic_height = 40;
    const float basic_text_size = 30;
    const float small_text_size = 26;
    const float very_small_text_size = 22;
    const float tiny_text_size = 20;
    const float tinier_text_size = 16;
    const float top_margin = 20;

    const float small_half_button_width = (basic_width - basic_margin) / 2;
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
        float y_offset;             // Offset modification on Y-axis
    };

    // Long configs
    std::string desc_label_text = "Ctrl = " + std::to_string((int)ctrl_speed) + " | Shift = " + std::to_string((int)shift_speed);

    // UI element definitions
    std::vector<UIElementConfig> element_configs = 
    {
        {"pause", "button", "Start", very_small_text_size, half_button_width, 0, true, 0},
        {"reset", "button", "Reset", very_small_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"prob_input", "text_area", "not set", small_text_size, half_button_width * 0.8f, 0, true, 0},
        {"generate", "button", "Generate", very_small_text_size, half_button_width * 1.2f, half_button_width * 0.8f + half_button_gap, false, 0},
        {"fps_label", "label", "not set", very_small_text_size, ui_width, -basic_margin, true, -15},
        {"iter_label", "label", "Iteration: 0", very_small_text_size, ui_width, -basic_margin, true, -40},
        {"speed_label", "label", "not set", very_small_text_size, ui_width, -basic_margin, true, -35},
        {"desc_label", "label", desc_label_text, very_small_text_size, ui_width, -basic_margin, true, -40},
        {"slower", "button", "<", basic_text_size, half_button_width, 0,  true, -30},
        {"faster", "button", ">", basic_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"outline", "button", "Grid: OFF", tiny_text_size,  half_button_width, 0, true, 0},
        {"streamline_button", "button", "Streamlines:\nNA", tinier_text_size,  half_button_width, half_button_width + half_button_gap, false , 0},
        {"vx_window_button", "button", "Show X\nvelocity", tinier_text_size, half_button_width, 0, true, 0},
        {"vy_window_button", "button", "Show Y\nvelocity", tinier_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"toggle_pu", "button", "not set", tiny_text_size,  half_button_width, 0, true , 0},
        {"save_as_img_button", "button", "Save Img", tiny_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"save_automaton_button", "button", "Save", very_small_text_size, half_button_width, 0, true, 0},
        {"load_automaton_button", "button", "Load", very_small_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"air_button", "button", "Air", very_small_text_size, half_button_width, 0, true, 15},
        {"gas_button", "button", "Fluid", very_small_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
        {"wall_button", "button", "Wall", very_small_text_size, half_button_width, 0, true, 0},
        {"cell_log_button", "button", "Unlock log", tinier_text_size, half_button_width, half_button_width + half_button_gap, false, 0},
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
            widgets[config.name] = button;
            gui_ref.add(button);
        }
        else if (config.element_type == "label") 
        {
            auto label = tgui::Label::create(config.text);
            label->setSize(config.width, basic_height);
            label->setTextSize(config.text_size);
            label->setHorizontalAlignment(tgui::Label::HorizontalAlignment::Center);
            label->setPosition(total_x_offset + config.x_offset, current_y_pos);
            widgets[config.name] = label;
            gui_ref.add(label);
        }
        else if (config.element_type == "spin_button")
        {
            auto spin_button = tgui::SpinButton::create(0.f, 1.f);
            spin_button->setSize(config.width, basic_height);
            spin_button->setTextSize(config.text_size);
            spin_button->setPosition(total_x_offset + config.x_offset, current_y_pos);
            widgets[config.name] = spin_button;
            gui_ref.add(spin_button);
        }
        else if (config.element_type == "text_area")
        {
            auto text_area = tgui::TextArea::create();
            text_area->setSize(config.width, basic_height);
            text_area->setTextSize(config.text_size);
            text_area->setText(config.text);
            text_area->setPosition(total_x_offset + config.x_offset, current_y_pos);
            widgets[config.name] = text_area;
            gui_ref.add(text_area);
        }
    }
}

/* Getters */
tgui::Widget::Ptr UI::get_widget(const std::string& name) 
{
    return widgets[name];
}

void UI::set_speed_label_speed(float speed)
{
    auto speed_label = get_widget_as<tgui::Label>("speed_label");
    std::string text = "Speed: " + std::to_string((int)speed) + " UPS";
    speed_label->setText(text);
}

void UI::set_fps_label_fps(int fps)
{
    auto fps_label = get_widget_as<tgui::Label>("fps_label");
    std::string text = "FPS: " + std::to_string(fps);
    fps_label->setText(text);
}

void UI::set_iteration_label_num(int iteration)
{
    auto iter_label = get_widget_as<tgui::Label>("iter_label");
    std::string text = "Iteration: " + std::to_string(iteration);
    iter_label->setText(text);
}