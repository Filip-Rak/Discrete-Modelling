import numpy
import pygame
from Automaton import CellularAutomaton, CellState, WindDirection

class Visualization:
    def __init__(self, automaton, cell_size=20):
        self.automaton = automaton
        self.cell_size = cell_size
        self.window_width = automaton.cols * cell_size + 250
        self.window_height = automaton.rows * cell_size

        # Update speed
        self.MAX_UPDATE_SPEED = 30
        self.MIN_UPDATE_SPEED = 1
        self.SPEED_CHANGE_SLOW = 1
        self.SPEED_CHANGE_NORMAL = 5
        self.SPEED_CHANGE_FAST = 10
        self.set_update_speed(5)
        self.time_since_last_update = 0;

        # Time
        self.delta_time = 0

        self.running = True
        self.paused = True
        self.selected_tool = CellState.EMPTY

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Forest Fire")
        self.clock = pygame.time.Clock()

        # Button positions
        WIDTH = 200
        HEIGHT = 40

        self.buttons = {
            "pause": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 25, WIDTH, HEIGHT), "text": "Start"},
            "reset": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 75, WIDTH, HEIGHT), "text": "Reset"},
            "slower": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 125, WIDTH / 2 - 10, HEIGHT), "text": "<<"},
            "faster": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20 + 110, 125, WIDTH / 2 - 10, HEIGHT), "text": ">>"},
            "wind": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 250, WIDTH, HEIGHT),"text": "Wind: %s" % self.automaton.wind_direction.name},
            "empty": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 300, WIDTH, HEIGHT), "text": "Ground"},
            "forest": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 350, WIDTH, HEIGHT), "text": "Forest"},
            "dense_forest": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 400, WIDTH, HEIGHT), "text": "Dense Forest"},
            "fire": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 450, WIDTH, HEIGHT), "text": "Fire"},
            "water": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 500, WIDTH, HEIGHT), "text": "Water"},
            "flood": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 550, WIDTH, HEIGHT), "text": "Flood"},
            "burned": {"rect": pygame.Rect(self.automaton.cols * self.cell_size + 20, 600, WIDTH, HEIGHT), "text": "Scorched"}
        }

    def draw_grid(self):
        """Draw the cellular automaton grid."""
        for row in range(self.automaton.rows):
            for col in range(self.automaton.cols):
                state = self.automaton.grid[row, col]
                color = self.get_color(CellState(state))
                pygame.draw.rect(self.screen, color,
                                 (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (200, 200, 200),  # Grid lines
                                 (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)

    def draw_buttons(self):
        """Draw the buttons."""
        font = pygame.font.Font(None, 36)
        for key, button in self.buttons.items():
            pygame.draw.rect(self.screen, (0, 0, 0), button["rect"])  # Button background
            text = font.render(button["text"], True, (255, 255, 255))
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)

    def draw_text(self):
        # Update Speed
        speed_font = pygame.font.Font(None, 32)
        speed_text = "Speed: %d UPS" % self.updates_per_sec
        speed_text_surface = speed_font.render(speed_text, True, (0, 0, 0))
        speed_text_rect = speed_text_surface.get_rect(center=(self.automaton.cols * self.cell_size + 95, 185))
        self.screen.blit(speed_text_surface, speed_text_rect)

        # Update speed key tooltip
        st_font = pygame.font.Font(None, 24)
        st_text = "Ctrl = %d | Shift = %d" % (self.SPEED_CHANGE_NORMAL, self.SPEED_CHANGE_FAST)
        st_text_surface = st_font.render(st_text, True, (0, 0, 0))
        st_text_rect = st_text_surface.get_rect(center=(self.automaton.cols * self.cell_size + 95, 205))
        self.screen.blit(st_text_surface, st_text_rect)

    def get_color(self, state):
        """Get color based on cell state."""
        color_map = {
            CellState.EMPTY: (153, 76, 0),
            CellState.FIRE: (255, 0, 0),
            CellState.WATER: (51, 153, 255),
            CellState.FLOOD: (102, 178, 255),
            CellState.FOREST: (0, 255, 0),
            CellState.OVERGORWN_FOREST: (0, 190, 0),
            CellState.BURNED: (160, 160, 160),
        }
        return color_map[state]

    def set_update_speed(self, new_speed):
        self.updates_per_sec = self.clamp(new_speed, self.MIN_UPDATE_SPEED, self.MAX_UPDATE_SPEED)
        self.time_between_updates = (1 / self.updates_per_sec) * 1000.0;    # Miliseconds

    def adjust_speed_change(self):
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_CTRL:
            return self.SPEED_CHANGE_NORMAL
        elif mods & pygame.KMOD_SHIFT:
            return self.SPEED_CHANGE_FAST

        return self.SPEED_CHANGE_SLOW

    def clamp(self, val, min, max):
        if val > max: return max
        if val < min: return min

        return val

    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Check if click is on the grid
                if x < self.automaton.cols * self.cell_size:
                    col, row = x // self.cell_size, y // self.cell_size
                    if 0 <= row < self.automaton.rows and 0 <= col < self.automaton.cols:
                        self.automaton.put_cell(row, col, self.selected_tool.value) # Set a new cell state
                # Check if click is on a button
                for key, button in self.buttons.items():
                    if button["rect"].collidepoint(x, y):
                        if key == "pause":
                            self.paused = not self.paused
                            self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                        elif key == "reset":
                            self.buttons["pause"]["text"] = "Start"
                            self.paused = True
                            print("Reset Clicked")
                            self.automaton.reset()
                        elif key == "slower":
                            speed_step = self.adjust_speed_change()
                            self.set_update_speed(self.updates_per_sec - speed_step)
                        elif key == "faster":
                            speed_step = self.adjust_speed_change()
                            self.set_update_speed(self.updates_per_sec + speed_step)
                        elif key == "wind":
                            directions = list(WindDirection)
                            current_index = directions.index(self.automaton.wind_direction)
                            new_direction = directions[(current_index + 1) % len(directions)]
                            self.automaton.set_wind(new_direction)
                            self.buttons["wind"]["text"] = "Wind: %s" % new_direction.name
                        elif key == "empty":
                            self.selected_tool = CellState.EMPTY
                        elif key == "forest":
                            self.selected_tool = CellState.FOREST
                        elif key == "dense_forest":
                            self.selected_tool = CellState.OVERGORWN_FOREST
                        elif key == "fire":
                            self.selected_tool = CellState.FIRE
                        elif key == "water":
                            self.selected_tool = CellState.WATER
                        elif key == "flood":
                            self.selected_tool = CellState.FLOOD
                        elif key == "burned":
                            self.selected_tool = CellState.BURNED

            elif event.type == pygame.MOUSEMOTION:
                # Check if the left mouse button is pressed
                if pygame.mouse.get_pressed()[0]:  # [0] for left button
                    x, y = pygame.mouse.get_pos()
                    if x < self.automaton.cols * self.cell_size:
                        col, row = x // self.cell_size, y // self.cell_size
                        if 0 <= row < self.automaton.rows and 0 <= col < self.automaton.cols:
                            # self.automaton.grid[row, col] = self.selected_tool.value  # Set cell state
                            self.automaton.put_cell(row, col, self.selected_tool.value)


            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def run(self):
        """Main loop for the visualization."""
        while self.running:
            self.handle_events()
            if not self.paused:
                self.time_since_last_update += self.delta_time
                if self.time_since_last_update >= self.time_between_updates:
                    self.automaton.update()
                    self.time_since_last_update = 0

            self.screen.fill((220, 220, 220))
            self.draw_grid()
            self.draw_buttons()
            self.draw_text()
            pygame.display.flip()
            self.delta_time = self.clock.tick(60)

        pygame.quit()