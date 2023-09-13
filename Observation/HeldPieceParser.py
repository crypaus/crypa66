import numpy as np
import cv2

from Agent.PieceClassifier import PieceClassifier
from FrameGrabber.Frame import Frame
from Observation.BlockDetector import BlockDetector

from collections import namedtuple

HeldPiece = namedtuple("HeldPiece", ("piece", "is_held_disabled"))
class HeldPieceParser:
    def __init__(self, held_roi, block_length):
        self.held_roi = held_roi
        self.block_length = block_length
        self.held_piece = None
        self.is_held_disabled = False

        self.old_contours = None

    def construct_feature(self):
        return HeldPiece(self.held_piece, self.is_held_disabled)

    def process(self, preprocessed_frame):
        preprocessed_frame = preprocessed_frame.copy()
        frame = preprocessed_frame.frame.copy()

        held_msk = Frame.create_mask(frame, [self.held_roi])
        held_msk = cv2.bitwise_and(held_msk, held_msk, mask=preprocessed_frame.masks["greedy"])
        
        contours, _ = cv2.findContours(held_msk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[::-1]

        if contours:
            self.old_contours = contours[0]

            x,y,w,h = cv2.boundingRect(contours[0])
            piece_roi = ((x, y+h-1), (x+w-1, y))
            piece_crop = Frame(Frame.crop(frame, piece_roi))

            max_h = np.max(piece_crop.hsv.h)
            max_s = np.max(piece_crop.hsv.s)
            max_v = np.max(piece_crop.hsv.v)
            self.is_held_disabled = (max_h == 0) and (max_s == 0) and (max_v < 70)

            block_detector = BlockDetector(piece_roi, self.block_length)
            blocks = block_detector.estimate_block_locations(held_msk)

            if blocks.size != 0:
                self.held_piece = PieceClassifier().match_piece(blocks)
                if self.held_piece:
                    self.held_piece.mark_as_held()

        # self.visualize_held_parser(frame)

    def visualize_held_parser(self, frame):
        if self.old_contours is not None:
            x,y,w,h = cv2.boundingRect(self.old_contours)
            held_parser_vis = cv2.rectangle(frame.copy(), (x, y+h-1), (x+w-1, y), (255, 0, 128), 1)
            cv2.imshow("held_parser_vis", held_parser_vis)
