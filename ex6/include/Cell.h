#pragma once

#include <cstdint>

class Cell
{
private:
	/* Attributes */
	uint8_t state;	// 8-bits, lower 4 is input, upper 4 is output

public:
	/* Constructor */
	Cell();

	/* Public Methods */
	enum Direction
	{
		UP = 0,
		RIGHT = 1,
		LEFT = 2,
		DOWN = 3
	};

	// Sets the cell to 0
	void reset();

	/* Private Methods */
	void change_state(Direction dir, bool value, int offset);

	/* Setters */
	void set_input(Direction dir, bool value);
	void set_output(Direction dir, bool value);

	/* Getters */
	// !-- bool get_state?
	bool is_active() const;
};