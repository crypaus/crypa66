import numpy as np
import cv2

from Observation.RoiEstimator import RoiEstimator
from Observation.ImageProc import ImageProc

from FrameGrabber.Frame import Frame

import Helpers.Geometry as Geo
from Helpers.Draw import Draw
from Types.Singleton import Singleton


class ContainerDetector(metaclass=Singleton):

    def __init__(self):
        self.container_lines = []
        self.previous_container_lines_array = np.array([])
        self.roi_estimator = RoiEstimator()
        self.previous_frame = None

    def has_feature(self):
        container_lines, rois = self.construct_feature()
        return (len(container_lines) != 0) and (rois is not None)

    def construct_feature(self):
        if len(self.previous_container_lines_array) == 0:
            return ([], None)

        container_lines = self.previous_container_lines_array
        rois = self.roi_estimator.rois
        # self.visualize_container_lines()
        return (container_lines, rois)

    def visualize_container_lines(self):
        if self.previous_frame is None:
            return None

        overlay = np.zeros_like(self.previous_frame)

        ## draw overlay
        draw = Draw.begin(overlay)
        for roi in self.roi_estimator.rois.values():
            Draw.rectangle(draw, roi, line_color=(50, 89, 255), thickness=2)
        Draw.lines(draw, np.array(self.container_lines), line_color=(50, 89, 168), thickness=3)
        overlay = Draw.end(draw)
        ## end

        vis = cv2.addWeighted(self.previous_frame, 0.2, overlay, 0.8, 0)
        cv2.imshow("container vis", Frame.resize(vis, 200))

    def refine_container_lines(self, container_lines):
        container_lines = np.array(container_lines)
        for line in container_lines:
            p1, p2 = line
            p1[Geo.Point.y] -= 1
            p1[Geo.Point.x] -= 2

            p2[Geo.Point.y] -= 1
            p2[Geo.Point.x] += 1
        return container_lines

    def process(self, preprocessed_frames):
        accumulated_lines = []
        self.previous_frame = preprocessed_frames[-1].frame
        y_max = Frame(self.previous_frame).height


        for frame in preprocessed_frames:
            ## Accumulate the container's horizontal lines (per frame)
            # masks = ImageProc.segment(frame)
            # masks = PerformanceChecker.check_performance(ImageProc.segment, frame)
            lines = ImageProc.detect_lines(frame.masks["container"])
            if len(lines) < 3:
                continue

            lines = self.filter_lines(lines, y_max)
            if self.lines_match_container_shape(lines, y_max):
                accumulated_lines.append(lines)

        if len(self.container_lines) * len(accumulated_lines) > 0:
            old_container_base = self.container_lines[-1]
            new_container_base = accumulated_lines[0][-1]
            container_deviation = Geo.Line.y_distance(old_container_base, new_container_base)
            if container_deviation < 5:
                accumulated_lines.insert(0, self.container_lines)

        accumulated_lines = np.array(accumulated_lines)
        self.container_lines = self.find_longest_lines(accumulated_lines)

        if len(self.container_lines) == 3:
            refined_container_lines = self.refine_container_lines(self.container_lines)
            self.roi_estimator.update_estimations(refined_container_lines)

        self.previous_container_lines_array = self.refine_container_lines(self.container_lines)


    def sort_lines_by_ypos(self, lines, ascending=True):
        key_y_position = lambda l: (l[0][Geo.Point.y] + l[1][Geo.Point.y]) // 2
        return sorted(lines, key=key_y_position, reverse=(not ascending))
    
    def filter_lines(self, lines, y_max):
        if len(lines) == 0:
            return []

        ## Remove lines with low verticle distance from other lines
        lines = self.filter_stacked_lines(lines)

        ## Remove lines nearly touching the bottom of the image
        lines = self.sort_lines_by_ypos(lines, ascending=True)
        good_lines = [l for l in lines if y_max - l[0][1] > 15]

        ## Take the bottom-most 3 lines
        return good_lines[-3:]

    def filter_stacked_lines(self, lines):
        lines = self.sort_lines_by_ypos(lines, ascending=True)

        previous_line = lines[0]
        good_lines = [previous_line]
        for line in lines[1:]:
            y_dist = Geo.Line.y_distance(line, previous_line)
            previous_line = line

            if y_dist > 20:
                ## Select lines that are far enough from others 
                good_lines.append(line)

            else:
                ## Make sure to not throw out any "fuller"/longer line segments
                line_length = Geo.Line.length(line)
                prev_length = Geo.Line.length(previous_line)
                if line_length > prev_length:
                    good_lines.pop(-1)
                    good_lines.append(line)

        return good_lines

    def find_longest_lines(self, accumulated_lines):
        if len(accumulated_lines) == 0:
            return []

        best_lines = []
        for i in range(0, 3):
            lines = accumulated_lines[:, i]
    
            best_line = lines[0]
            best_length = Geo.Line.length(best_line)

            for line in lines[1:]:
                line_length = Geo.Line.length(line)
                if line_length > best_length:
                    best_length = line_length
                    best_line = line

            best_lines.append(best_line.tolist())

        return best_lines

    def lines_match_container_shape(self, lines, y_max):
        if len(lines) < 3:
            return False

        lines = self.sort_lines_by_ypos(lines, ascending=True)

        top_most_line = lines[0]
        middle_line = lines[1]
        bottom_most_line = lines[2]

        top_to_middle_dist = Geo.Line.y_distance(top_most_line, middle_line) / y_max
        c1 = top_to_middle_dist < 0.55

        top_to_bottom_dist = Geo.Line.y_distance(top_most_line, bottom_most_line) / y_max
        c2 = top_to_bottom_dist > 0.45

        middle_to_bottom_dist = Geo.Line.y_distance(middle_line, bottom_most_line) / y_max
        c3 = middle_to_bottom_dist < 0.2

        top_length = Geo.Line.length(top_most_line)
        bottom_length = Geo.Line.length(bottom_most_line)
        bottom_to_top_ratio = bottom_length / top_length
        c4 = bottom_to_top_ratio > 1.8

        return all([c1, c2, c3, c4])
