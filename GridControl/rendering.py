import pyglet
from pyglet.window import mouse
from pyglet.window import key
import numpy as np

def draw_circle(x, y, color, radius):
    num_sides = 50
    verts = [x, y]
    colors = list(color)
    for i in range(num_sides + 1):
        verts.append(x + radius * np.cos(i * np.pi * 2 / num_sides))
        verts.append(y + radius * np.sin(i * np.pi * 2 / num_sides))
        colors.extend(color)
    pyglet.graphics.draw(len(verts) // 2, pyglet.gl.GL_TRIANGLE_FAN,
                        ('v2f', verts), ('c3f', colors))


def render(device_list, update):
        screen =  pyglet.canvas.get_display().get_default_screen()
        wS = int(min(screen.width, screen.height) * 1 / 3)


        window = pyglet.window.Window(wS, wS)
        # Set Cursor
        cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
        window.set_mouse_cursor(cursor)

        # Outlines
        lower_grid_coord = wS * 0.075
        board_size = wS * 0.85
        upper_grid_coord = board_size + lower_grid_coord

        @window.event
        def on_draw():
            pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
            window.clear()

            pyglet.gl.glLineWidth(3)
            for device in device_list:
                x = int(device.position[0]*wS/2 + wS/2)
                y =int(wS/2 - device.position[1]*wS/2)
                #print(f"wS {wS}; id {device.id}; x,y {(x ,y)}; pos {device.position}")
                draw_circle(x, y, [0.05882352963, 0.180392161, 0.2470588237], 5)

        pyglet.clock.schedule_interval(update, 1/120.0)
        pyglet.app.run()