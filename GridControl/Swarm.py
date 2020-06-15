import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np
from Device import Device
class Swarm:
    def __init__(self, cell_nb=5):
        self.dList = []
        self.cell_nb = cell_nb
        pass
    
    def discretize(self):
        """
        Creates the frequency map from the list of devices and the number of cells.
        """
        assert(self.dList is not [], "Devices not initialized, call dList_init before creating the frequency map.")
        
        f_map = np.zeros((self.cell_nb, self.cell_nb))

        for dev in self.dList:
            x, y = dev.position
            
            if x >= 1 : 
                x = 0.999
            elif x < -1:
                x = -1
            elif y >= 1:
                y = 0.999
            elif y < -1:
                y = -1

            cx = int(np.floor((x+1)*(self.cell_nb)/2))
            cy = int(np.floor((y+1)*(self.cell_nb)/2))
            print(f"cx, cy : {(cx, cy)} ; x, y : {(x, y)}")
            assert((cx>0 and cx<=self.cell_nb-1 and cy>0 and cy<=self.cell_nb-1), f"Device {dev.id} out of bound (position tuple {(dev.position[0], dev.position[0])}).")
            
            f_map[cy][cx] += 1
        
        return f_map


    def dList_init(self, initial_conditions):
        """
        Initializes and instanciates all devices given their initial conditions; 
        initial_conditions is of the form [(x_0, y_0), (vx_0, vy_0), ..., (x_n, y_n), (vx_n, vy_n)] describing the initial parameters all devices (from 0 to n)
        """

        for dID, (pos, vel) in enumerate(initial_conditions):
            self.dList.append(Device(dID, pos, vel))
        
        return self.dList

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