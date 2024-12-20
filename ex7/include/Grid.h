#pragma once

#include <cstring>

/* Structs */
class Grid
{
public:
	/* Statics & Constants */
	static constexpr int direction_num = 4;	// Left, Top, Right, Down
	static constexpr int directions_x[direction_num] = { -1, 0, 1, 0 };
	static constexpr int directions_y[direction_num] = { 0, 1, 0, -1 };
	static constexpr int opposite_directions[direction_num] = { 2, 3, 0, 1 };
	static constexpr double weights[direction_num] = { 0.25, 0.25, 0.25, 0.25 };

private:
	/* Attributes */
	// Dimensions
	int width;
	int height;

	// Modifiers
	double tau = 1.5f;

	// Arrays for cell data
	double* concentration;	// 0.0 - 1.0
	bool* is_wall;			// Treated as impassable by gas

	// Indexing: [direction][cell_num]
	double* f_in[direction_num];	// Input functions
	double* f_eq[direction_num];	// Equlibrium distibution functions
	double* f_out[direction_num];	// Output functions

public:
	/* Frenship Declaration */
	friend class Automaton;

	/* Constructors */
	Grid(int w, int h);

	// Copy constructor (deep copy)
	Grid(const Grid& other);

	/* Destructor */
	~Grid();

	/* Static Methods */
	int get_id(int x_pos, int y_pos);

	/* Setters */
	void set_cell_as_active(int x, int y);
	void set_cell_as_active(int cell_id);

	void set_cell_as_inactive(int x, int y);
	void set_cell_as_inactive(int cell_id);

	void set_cell_as_wall(int x, int y);
	void set_cell_as_wall(int cell_id);

	/* Getters */
	double get_cell_concetration(int x, int y);
	double get_cell_concetration(int cell_id);

	bool get_cell_is_wall(int x, int y);
	bool get_cell_is_wall(int cell_id);
};