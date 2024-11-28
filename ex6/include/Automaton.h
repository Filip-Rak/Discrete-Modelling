#pragma once

#include <cstdint>
#include <cstdlib>

class Automaton
{
public:
	/* Enums */
	enum Direction
	{
		UP = 0,		// Bit 0
		RIGHT = 1,	// Bit 1
		LEFT = 2,	// Bit 2
		DOWN = 3	// Bit 3
	};

	enum State
	{
		WALL = 0b00 << 14,	// Encoded as 00 in bits 15-14
		EMPTY = 0b01 << 14,
		GAS = 0b10 << 14
	};

private:
	/* Attributes */
	uint16_t* cells;
	int width;
	int height;

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Grid Operations */
	void generate_random(float probability = 0.5f);	// Creates a random grid
	void update();
	void reset();

	/* Accessors */
	inline State get_state(uint16_t cell)
	{
		return static_cast<State>(cell & 0b1100000000000000); // Mask bits 15-14
	}

	inline uint16_t set_state(uint16_t cell, State state) 
	{
		return (cell & 0b0011111111111111) | state; // Clear bits 15-14, then set new state
	}

	// Input directions (lower 4 bits)
	inline uint8_t get_input(uint16_t cell) 
	{
		return cell & 0x0F;
	}

	inline uint16_t set_input(uint16_t cell, uint8_t input) 
	{
		return (cell & 0xFFF0) | (input & 0x0F); // Clear lower 4 bits, then set input
	}

	// Output directions (next 4 bits)
	inline uint8_t get_output(uint16_t cell) 
	{
		return (cell >> 4) & 0x0F;
	}

	inline uint16_t set_output(uint16_t cell, uint8_t output) 
	{
		return (cell & 0xFF0F) | ((output & 0x0F) << 4); // Clear bits 4-7, then set output
	}
};
