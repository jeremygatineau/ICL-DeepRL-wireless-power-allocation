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


def render(device_list, nb_cell, f_map):
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
            batch = pyglet.graphics.Batch()
            draw_grid(batch, wS, nb_cell)
            draw_count(batch, f_map, wS, nb_cell)
            pyglet.gl.glLineWidth(3)
            for device in device_list:
                x = int(device.position[0]*wS/2 + wS/2)
                y =int(wS/2 - device.position[1]*wS/2)
                #print(f"wS {wS}; id {device.id}; x,y {(x ,y)}; pos {device.position}")
                draw_circle(x, y, [0.05882352963, 0.180392161, 0.2470588237], 5)

            batch.draw()
        #pyglet.clock.schedule_interval(update, 1/120.0)
        pyglet.app.run()

def draw_grid(batch, wS, nb_cell):
    label_offset = 20
    left_coord = 0
    right_coord = 0
    ver_list = []
    color_list = []
    num_vert = 0
    for i in range(nb_cell):
        # horizontal
        ver_list.extend((0, left_coord,
                         wS, right_coord))
        # vertical
        ver_list.extend((left_coord, 0,
                         right_coord, wS))
        color_list.extend([0.3, 0.3, 0.3] * 4)  # black

        left_coord += wS/nb_cell
        right_coord += wS/nb_cell
        num_vert += 4
    batch.add(num_vert, pyglet.gl.GL_LINES, None,
              ('v2f/static', ver_list), ('c3f/static', color_list))

def draw_count(batch, f_map, wS, cell_nb):
    print(f_map)
    n = len(f_map)
    for i in range(n):
        for j in range(n):
            pyglet.text.Label(str(f_map[i, j]),
                        font_name='Courier', font_size=11,
                          x=(j+1/2)*wS/cell_nb, y=((n-i-1)+1/2)*wS/cell_nb,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 255), batch=batch)