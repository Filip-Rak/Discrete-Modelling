#pragma once

#include <cstring>

/* Structs */
class Grid
{
private:
	/* Statics & Constants */
	static const int direction_num = 4;

	/* Attributes */

	// Dimensions
	int width;
	int height;

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

	/* Setters */
	void set_cell_as_active(int x, int y);
	void set_cell_as_inactive(int x, int y);
	void set_cell_as_wall(int x, int y);

	/* Getters */
	double get_cell_concetration(int x, int y);
	bool get_cell_is_wall(int x, int y);
};