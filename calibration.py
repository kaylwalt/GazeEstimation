#This code is original
#
#
import pygame as pg
import math
import csv
import os
import datetime
from random import *
from cap import *

csvFile = "eyeposition.csv"
if not os.path.isfile(csvFile):
    print("making new csv file")
    with open(csvFile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['xpixel', 'ypixel', 'xestimate', 'yestimate', 'time'])

red = (255, 0, 0)
blue = (0, 0, 255)
magenta = (255, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)

def calibrate():
    pg.init()
    rr = []
    bc = white
    myfont = pg.font.SysFont(pg.font.get_default_font(), 35)
    clock = pg.time.Clock()
    display = pg.display.set_mode((0,0), pg.FULLSCREEN)
    rr += [display.fill(bc)]
    #rr += [write_message(display, "We are going to calibrate your screen location", myfont, white)]
    test_data = []
    test_points = [(0,0), (display.get_width(), 0), (0, display.get_height()), (display.get_width(), display.get_height())] * 3
    current_point = testPoint(test_points[0])
    test_points = test_points[1:]
    rr += [screen_prompt_cal(display, current_point, myfont, white)]
    running = True
    since_last_frame = 0
    while running:
        for e in pg.event.get():
            if e.type is pg.QUIT:
                running = False
            elif e.type is pg.KEYDOWN:
                if e.key is pg.K_ESCAPE:
                    running = False
                if e.key is pg.K_SPACE:
                    if current_point.valid():
                        test_data += [[current_point.point, (0,0)]]
                        rr += [display.fill(bc, current_point.corner_update(display, white))]
                        if test_points == []:
                            running = False
                        else:
                            current_point = testPoint(test_points[0])
                            test_points = test_points[1:]
                            rr += [screen_prompt_cal(display, current_point, myfont, white)]

        #update as close to 60 fps as possible
        since_last_frame += clock.tick()
        if since_last_frame < 1000.0/60.0:
            continue
        rr += [current_point.corner_update(display, white)]
        pg.display.update(rr)
        rr = []
        since_last_frame = 0
    pg.quit()
    print(test_data)

def main():
    pg.init()
    redraw_recs = []
    myfont = pg.font.SysFont(pg.font.get_default_font(), 30)
    clock = pg.time.Clock()
    dis_surf = pg.display.set_mode((0,0), pg.FULLSCREEN)
    fullscreen = True
    background_color = white
    redraw_recs += [dis_surf.fill(background_color)]
    running = True
    time = 0
    test_point = testPoint((30, 30), 20, 3)
    testing = True
    while running:
        time += clock.tick()
        if time < 1000.0/60.0:
            continue
        if time > 0:
            redraw_recs += [write_message(dis_surf, "%9.5f fps" % (1000.0/time), myfont, background_color)]
            time = 0

        if testing:
            rec = test_point.update(dis_surf, background_color)
            if rec is not None:
                redraw_recs += [rec]

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    if testing and test_point.valid():
                        data = [test_point.point[0], test_point.point[1], 0, 0, datetime.datetime.now()]
                        print(data)
                        redraw_recs += [dis_surf.fill(background_color, test_point.rect)]
                        write_eye_position(csvFile, data)
                        test_point = randomPoint(dis_surf)

                if event.key == pg.K_ESCAPE:
                    running = False
                if event.key == pg.K_f:
                    if fullscreen:
                        dis_surf = pg.display.set_mode((0,0), pg.RESIZABLE)
                        fullscreen = False
                        redraw_recs += [dis_surf.fill(background_color)]
                    else:
                        dis_surf = pg.display.set_mode((0,0), pg.FULLSCREEN)
                        fullscreen = True
                        redraw_recs += [dis_surf.fill(background_color)]

        pg.display.update(redraw_recs)
        redraw_recs = []
    pg.quit()


def write_eye_position(filename, data):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def screen_prompt_cal(surface, testpoint, font, color):
    if testpoint.point == (0, 0):
        return write_message(surface, "testing top left", font, color)
    elif testpoint.point == (0, surface.get_height()):
        return write_message(surface, "testing bottom left", font, color)
    elif testpoint.point == (surface.get_width(), 0):
        return write_message(surface, "testing top right", font, color)
    elif testpoint.point == (surface.get_width(), surface.get_height()):
        return write_message(surface, "testing bottom right", font, color)
    else:
        print("messsed up calibration")
        print(testpoint.point)
        return write_message(surface, "this is the wrong test point", font, color)

def write_message(surface, message, font, redraw_color):
    width = surface.get_width()
    height = surface.get_height()
    textsurface = font.render(message, False, blue)
    textsize = textsurface.get_size()
    cover = pg.Surface(textsize)
    cover.fill(redraw_color)
    #print("width {}, height {}".format(width, height))
    #print("textsize {}".format(textsize))
    placement = (width/2 - textsize[0]/2, height/2 - textsize[1]/2)
    #print("placement {}".format(placement))
    surface.blit(cover, placement)
    drawn_rect = surface.blit(textsurface, placement)
    return drawn_rect

class testPoint():
    #assuming 60fps
    def __init__(self, point, init_radius=20, error_margin=2):
        self.frames_since_shrink = 0
        self.point = point
        self.radius = init_radius
        self.init_radius = init_radius
        self.error_margin = error_margin
        self.shrink_per_second = 10
        self.shrink = True
        self.rect = pg.Rect(point[0] - self.init_radius, point[1] - self.init_radius, 2 * self.init_radius, 2 * self.init_radius)

    def update(self, surface, color):
        if self.frames_since_shrink < int(60/self.shrink_per_second):
            self.frames_since_shrink += 1
            return None
        else:
            self.frames_since_shrink = 0
            if self.radius <= 4 + 2:
                self.shrink = False
            if self.radius >= self.init_radius:
                self.shrink = True
            if self.shrink:
                self.radius -= 1
            else:
                self.radius += 1

            surface.fill(color, self.rect)
            pg.draw.circle(surface, blue, self.point, self.radius, 2)
            pg.draw.circle(surface, red, self.point, 4)
            return self.rect

    def corner_update(self, surface, color):
        if self.frames_since_shrink < int(60/self.shrink_per_second):
            self.frames_since_shrink += 1
            return None
        else:
            self.frames_since_shrink = 0
            if self.radius <= 4 + 2:
                self.shrink = False
            if self.radius >= self.init_radius:
                self.shrink = True
            if self.shrink:
                self.radius -= 1
            else:
                self.radius += 1

            offset = [0, 0]
            if self.point == (0, 0):
                offset = [4, 4]
            elif self.point == (surface.get_width(), 0):
                offset = [-4, 4]
            elif self.point == (0, surface.get_height()):
                offset = [4, -4]
            elif self.point == (surface.get_width(), surface.get_height()):
                offset = [-4, -4]

            surface.fill(color, self.rect.move(offset[0], offset[1]))
            pg.draw.circle(surface, blue, (self.point[0] + offset[0], self.point[1] + offset[1]), self.radius, 2)
            pg.draw.circle(surface, red, (self.point[0] + offset[0], self.point[1] + offset[1]), 4)
            return self.rect.move(offset[0], offset[1])

    def valid(self):
        #one second mess up allowance
        return (self.radius < 4 + 2 + self.error_margin)

def randomPoint(surface):
    width = surface.get_width()
    height = surface.get_height()
    return testPoint((randint(0, width-1), randint(0, height-1)), 20, 3)

if __name__ == "__main__":
    #main()
    calibrate()
