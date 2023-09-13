import numpy as np
import math
import cv2

from FrameGrabber.Frame import Frame

from Helpers.Draw import Draw
import Helpers.Geometry as Geo


class BlockDetector:
    def __init__(self, roi, block_length):
        roi_width = Geo.Rectangle.width(roi)
        roi_height = Geo.Rectangle.height(roi)
        self.grid_width = int(round(roi_width / block_length, 0))
        self.grid_height = int(round(roi_height / block_length, 0))

        self.blocks = np.full((self.grid_height, self.grid_width), 0, dtype=np.int8)
        self.block_length = block_length
        self.current_roi = roi
        self.probe_points = self.calculate_probe_points()
        self.current_frame = None

    def process(self, preprocessed_frame):
        self.current_frame = preprocessed_frame.frame.copy()

        roi_mask = Frame.create_mask(self.current_frame, [self.current_roi])
        # blocks_im = cv2.bitwise_and(self.current_frame, self.current_frame, mask=roi_mask)

        blocks_msk = preprocessed_frame.masks["blocks"]
        blocks_msk = cv2.bitwise_and(blocks_msk, blocks_msk, mask=roi_mask)

        # blocks_msk = ImageProc.segment_2(blocks_im)["blocks"]
        # blocks_msk = PerformanceChecker.check_performance(ImageProc.segment_2, blocks_im)["blocks"]

        self.blocks = self.estimate_block_locations(blocks_msk)
        # self.blocks = PerformanceChecker.check_performance(self.estimate_block_locations, blocks_msk)
        
    def calculate_probe_points(self):
        origin, _ = self.current_roi
        origin_x, origin_y = origin
        half_step = int(self.block_length) / 2

        points = []
        for y in range(self.grid_height, 0, -1):
            for x in range(1, self.grid_width+1):
                each_y = origin_y - (y * self.block_length)
                each_y += half_step #+ (half_step/5)
                each_y = max(2, each_y)

                each_x = origin_x + (x * self.block_length)
                each_x -= half_step

                points += [(each_x, each_y)]

        return points

    def estimate_block_locations(self, blocks_msk):
        ## probe for block locations
        self.blocks.fill(0)

        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                mask_idx = self.probe_points[x + (y * self.grid_width)]
                mask_idx = tuple(list(map(int, mask_idx))[::-1])
                self.blocks[y, x] = blocks_msk[mask_idx] > 0

        return self.blocks

    def visualize_block_detector(self, frame):
        color = (180, 253, 123)
        dframe = frame.copy()

        draw = Draw.begin(dframe)
        Draw.rectangle_grid(draw, self.current_roi, self.grid_width, (128, 128, 128))
        Draw.points(draw, self.probe_points, color)
        overlay = Draw.end(draw)

        vis = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        return vis

    def visualize_blocks(self, title, blocks, padding=0):
        frame = self.current_frame.copy()
        blocks_im = blocks.copy().ravel().repeat(3).reshape((self.grid_height, self.grid_width, 3))

        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                if blocks_im[y, x, 0] == 0:
                    continue
                frame_idx = self.probe_points[x + (y * self.grid_width)]
                frame_idx = tuple(list(map(int, frame_idx))[::-1])
                blocks_im[y, x] = frame[frame_idx]

        blocks_im = Frame.resize(blocks_im, 400)
        if padding != 0:
            padding_bar = np.full((blocks_im.shape[0], padding, 3), (128, 128, 128), np.uint8)
            blocks_im = np.hstack((padding_bar, blocks_im, padding_bar))
        cv2.imshow(title, blocks_im)

    def construct_feature(self):
        # cv2.imshow("block detector", self.visualize_block_detector(self.current_frame))
        return self.blocks
