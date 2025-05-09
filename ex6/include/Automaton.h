#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <bitset>
#include <iostream>

#include "AutomatonCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	// Components
	AutomatonCUDA cuda_helper;

	// Bit format //
	// 15 14 | 13 12 11 10 9 8 | 7 6 5 4 | 3 2 1 0
	// State |      Empty	   | Outputs | Inputs
	uint16_t* cells;
	uint16_t* cells_fallback;	
	int width;
	int height;

public:
	/* Constructor & Destructor */
	Automaton(int width, int height);
	~Automaton();

	/* Grid Operations */
	void generate_random_legacy(float probability = 0.1f);	// Creates a random grid
	void generate_random(float probability = 0.1f);	// Creates a random grid
	void update_cpu();
	void update_gpu();
	void update(bool use_gpu);
	void reset();

	/* Getters */
	inline uint16_t* get_cells()
	{
		return this->cells;
	}

	inline int get_width() const
	{
		return width;
	}	
	
	inline int get_height() const
	{
		return height;
	}

	/* Static Accessors */
	__host__ __device__ inline static State get_state(uint16_t cell)
	{
		return static_cast<State>(cell & 0b1100000000000000); // Mask bits 15-14
	}

	__host__ __device__ inline static uint16_t set_state(uint16_t cell, State state)
	{
		return (cell & 0b0011111111111111) | state; // Clear bits 15-14, then set new state
	}

	// Input directions (lower 4 bits)
	__host__ __device__ inline static uint8_t get_input(uint16_t cell)
	{
		return cell & 0x0F;
	}

	__host__ __device__ inline static uint16_t set_input(uint16_t cell, uint8_t input)
	{
		return (cell & 0xFFF0) | (input & 0x0F); // Clear lower 4 bits, then set input
	}

	// Output directions (next 4 bits)
	__host__ __device__ inline static uint8_t get_output(uint16_t cell)
	{
		return (cell >> 4) & 0x0F;
	}

	__host__ __device__ inline static uint16_t set_output(uint16_t cell, uint8_t output)
	{
		return (cell & 0xFF0F) | ((output & 0x0F) << 4); // Clear bits 4-7, then set output
	}

private:
	/* Private Functions */
	void output_to(uint16_t* output_arr, int sender_id, int receiver_id, Direction forward_direction, Direction opposite_direction);
};
