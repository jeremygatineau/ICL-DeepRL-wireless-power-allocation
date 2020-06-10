import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np

class Swarm:
    def __init__(self):
        pass
    
    def render(self):

        screen =  pyglet.canvas.get_display().get_screen()
        window_width = int(min(screen.width, screen.height) * 2 / 3)
        window_height = int(window_width * 1.2)
        window = pyglet.window.Window(window_width, window_height)

        self.window = window
        self.pyglet = pyglet
        self.user_action = None

        # Set Cursor
        cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
        window.set_mouse_cursor(cursor)

        # Outlines
        lower_grid_coord = window_width * 0.075
        board_size = window_width * 0.85
        upper_grid_coord = board_size + lower_grid_coord
    
        @window.event
        def on_draw():
            pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
            window.clear()

            pyglet.gl.glLineWidth(3)
            batch = pyglet.graphics.Batch()

            #rendering.draw_title(batch, window_width, window_height)

            batch.draw()

            # draw the pieces
            #rendering.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state)
        
        pyglet.app.run()

        return