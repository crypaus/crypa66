import numpy as np
import math

import Helpers.Geometry as Geo


class RoiEstimator:
    def __init__(self):
        self.container_lines = None
        self.block_length = 0
        self.rois = {
            "sky" : ((0,0), (1,1)),
            "game": ((0,0), (1,1)),
            "next": ((0,0), (1,1)),
            "held": ((0,0), (1,1)),
        }

    def update_estimations(self, container_lines):
        self.container_lines = np.array(container_lines)

        gameboard_base = self.container_lines[-1]
        gameboard_width = Geo.Line.length(gameboard_base)
        self.block_length = (gameboard_width / 10) * 1.00675

        self.rois["game"] = self.estimate_game_roi()
        self.rois["sky"] = self.estimate_sky_roi()
        self.rois["next"] = self.estimate_next_roi()
        self.rois["held"] = self.estimate_held_roi()

    def truncate_roi(self, roi):
        origin    = tuple([math.floor(c) for c in roi[0]])
        top_right = tuple([math.floor(c) for c in roi[1]])

        return (origin, top_right)

    def estimate_sky_roi(self):
        origin = self.container_lines[2][0]
        origin[Geo.Point.y] -= self.block_length * 20

        width = self.block_length * 10
        height = self.block_length * 3

        top_right_x = origin[Geo.Point.x]+width
        top_right_y =  max(0, origin[Geo.Point.y]-height)
        top_right = (top_right_x, top_right_y)

        return self.truncate_roi((origin, top_right))

    def estimate_game_roi(self):
        origin = self.container_lines[2][0]

        width = self.block_length * 10
        height = self.block_length * 20
        top_right = (origin[Geo.Point.x]+width, origin[Geo.Point.y]-height)

        return self.truncate_roi((origin, top_right))

    def estimate_next_roi(self):
        container_next_line = self.container_lines[1].copy()
        origin = container_next_line[0]
        origin[Geo.Point.x] += 2
        origin[Geo.Point.y] -= 2

        bottom_right = container_next_line[1]
        top_right_x = bottom_right[Geo.Point.x]
        top_right_y = self.rois["game"][1][Geo.Point.y]
        top_right_y += self.block_length * 0.8
        top_right = (top_right_x, top_right_y)

        return self.truncate_roi((origin, top_right))

    def estimate_held_roi(self):
        container_held_line = self.container_lines[0].copy()
        origin = container_held_line[0]
        origin[Geo.Point.x] += 2
        origin[Geo.Point.y] -= 2

        bottom_right = container_held_line[1]
        top_right_x = bottom_right[Geo.Point.x]
        top_right_y = self.rois["game"][1][Geo.Point.y]
        top_right_y += self.block_length * 0.8
        top_right = (top_right_x, top_right_y)

        return self.truncate_roi((origin, top_right))
