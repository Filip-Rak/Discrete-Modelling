#pragma once

#include <cstring>
#include <iostream>
#include <sstream>

/* Structs */
class Grid
{
public:
	/* Statics & Constants */
    static constexpr int direction_num = 9; // D2Q9 model
    static constexpr int directions_x[direction_num] = { 0, 1, -1, 0, 0, 1, -1, -1, 1 };
    static constexpr int directions_y[direction_num] = { 0, 0, 0, 1, -1, 1, 1, -1, -1 };
    static constexpr int opposite_directions[direction_num] = { 0, 2, 1, 4, 3, 7, 8, 5, 6 };
    static constexpr double weights[direction_num] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                                                        1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                                                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
private:
	/* Attributes */
	// Dimensions
	int width;
	int height;

	// Modifiers
	double tau = 1.5f;

	// Arrays for cell data
	double* density;
	double* velocity_x;
	double* velocity_y;
	bool* is_wall;			// Treated as impassable by gas

	// Indexing: [direction][cell_num]
	double* f_in[direction_num];	// Input functions
	double* f_buffer[direction_num];	// Buffer for temporary functions

public:
	/* Frenship Declaration */
	friend class Automaton;

	/* Constructors */
	Grid(int w, int h);

	// Copy constructor (deep copy)
	Grid(const Grid& other);

	/* Destructor */
	~Grid();

	/* Public Methods */
	void print_cell_data(int cell_id, int iteration);

	/* Static Methods */
	int get_id(int x_pos, int y_pos);

	/* Setters */
	void set_cell_as_active(int x, int y, double density = 1);
	void set_cell_as_active(int cell_id, double density = 1);

	void set_cell_as_inactive(int x, int y);
	void set_cell_as_inactive(int cell_id);

	void set_cell_as_wall(int x, int y);
	void set_cell_as_wall(int cell_id);

	/* Getters */
	double get_cell_concetration(int x, int y);
	double get_cell_concetration(int cell_id);

	bool get_cell_is_wall(int x, int y);
	bool get_cell_is_wall(int cell_id);

	double get_velocity_x(int cell_id);
	double get_velocity_y(int cell_id);
};