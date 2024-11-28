#include "Cell.h"

/* Constructor */
Cell::Cell(): state(0) {}

/* Private Methods */
void Cell::change_state(Direction dir, bool value, int offset)
{
    if (value) 
    {
        state |= (1 << (dir + offset)); // Set to 1
    }
    else 
    {
        state &= ~(1 << (dir + offset));    // Set to 0
    }
}

/* Public Methods */
void Cell::reset()
{
    state = 0;
}

/* Setters */
void Cell::set_input(Direction dir, bool value)
{
    change_state(dir, value, 0);
}

void Cell::set_output(Direction dir, bool value)
{
    change_state(dir, value, 4);
}

bool Cell::is_active() const
{
    return (state & 0x0F) > 0;  // Check if first 4 bits are higher than 0
}
