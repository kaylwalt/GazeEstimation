#landmark detection implementation from https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#
#
import pygame as pg
import math
import csv
import os
import datetime
from random import *
from cap import *
import numpy as np
import tensorflow as tf
from helper_fun import *


for i in range(0, 100):
    if not os.path.isfile("data/eyeposition_{}.csv".format(i)):
        csvFile = "data/eyeposition_{}.csv".format(i)
        print("making new csv file")
        with open(csvFile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['xpixel', 'ypixel', 'xestimate', 'yestimate', 'time'])
        break
    else:
        print("data/eyeposition_{}.csv".format(i) + " was taken")

red = (255, 0, 0)
blue = (0, 0, 255)
magenta = (255, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)

def main():
    model_dir = "../model_medium_no_dropout_short_train/"
    # Create the Estimator
    # run_config = tf.estimator.RunConfig().replace(
    #   session_config=tf.ConfigProto(device_count={'GPU': 0}))
    print("making the classifier")
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)
    fast = FastPredict(classifier)
    print("initializing the graph")
    trash = fast.predict({"x": np.zeros((1,448,448,3), dtype="float32")})

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    camera_matrix = np.array(loadeddict.get('camera_matrix'))
    dist_coeff = np.array(loadeddict.get('dist_coeff'))
    face_model = np.load("face_model.npy")

    #print("camera matrix: ", camera_matrix)
    #print("Dis coeff: ", dist_coeff)
    #print("face model: ", face_model.T)

    detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    #set height
    cap.set(3, 1080)
    #set width
    cap.set(4, 1920)


    pg.init()
    rr = []
    bc = white
    myfont = pg.font.SysFont(pg.font.get_default_font(), 35)
    clock = pg.time.Clock()
    display = pg.display.set_mode((0,0), pg.FULLSCREEN)
    rr += [display.fill(bc)]
    test_data = []
    test_points = [(0,0), (display.get_width(), 0), (0, display.get_height()), (display.get_width(), display.get_height())] * 3
    current_point = testPoint(test_points[0])
    test_points = test_points[1:]
    #rr += [screen_prompt_cal(display, current_point, myfont, white)]
    running = True
    since_last_frame = 0
    sample_count = 0
    while running:
        for e in pg.event.get():
            if e.type is pg.QUIT:
                running = False
            elif e.type is pg.KEYDOWN:
                if e.key is pg.K_ESCAPE:
                    running = False
                if e.key is pg.K_SPACE:
                    try:
                        #vec is the vector in camera space pointing from face center to the gaze point
                        cam_dir_vec, face_Center = predict_gaze(fast, cap, detector, dlib_predictor, camera_matrix, dist_coeff, face_model)
                        p = LinePlaneCollision(np.array([0, 0, 1]), np.array([0,0,0]), cam_dir_vec, face_Center)
                        sample_count += 1
                        write_eye_position(csvFile, [0,0,p[0],p[1], datetime.datetime.now()])
                        rr += [write_message(display, "---------------" + "Sample count number {}".format(sample_count) + "---------------", myfont, white)]
                    except:
                        rr += [write_message(display, "------------- couldnt get your face -------------", myfont, white)]
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
    main()
