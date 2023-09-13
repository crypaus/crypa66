import cv2

from Agent.PieceClassifier import PieceClassifier
from FrameGrabber.Frame import Frame
from Observation.BlockDetector import BlockDetector


class PieceQueueParser:
    def __init__(self, queue_roi, block_length):
        self.queue_roi = queue_roi
        self.block_length = block_length
        self.piece_queue = []

        self.prev_contours = None

    def construct_feature(self):
        queue_length = len(self.piece_queue)
        if queue_length == 0:
            return None
        elif queue_length == 1:
            return self.piece_queue[0]
            
        return self.piece_queue

    def process(self, preprocessed_frame):
        preprocessed_frame = preprocessed_frame.copy()
        frame = preprocessed_frame.frame.copy()

        queue_msk = Frame.create_mask(frame, [self.queue_roi])
        queue_msk = cv2.bitwise_and(queue_msk, queue_msk, mask=preprocessed_frame.masks["blocks"])
        queue_piece_img = cv2.bitwise_and(frame, frame, mask=queue_msk)
        preprocessed_frame.frame = queue_piece_img

        contours, _ = cv2.findContours(queue_msk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_contours = contours

        self.piece_queue = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            block_roi = ((x,y+h), (x+w,y))

            block_detector = BlockDetector(block_roi, self.block_length)
            # block_detector.process(queue_piece_img)
            block_detector.process(preprocessed_frame)

            blocks = block_detector.construct_feature()
            if blocks.size != 0:
                each_piece = PieceClassifier().match_piece(blocks)
                if each_piece is not None:
                    self.piece_queue.insert(0, each_piece.piece_name)

    def visualize_queue_parser(self, frame):
        if (self.prev_contours is None):
            return None

        block_detector_vis = frame.copy()
        for cnt in self.prev_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            block_roi = ((x,y+h), (x+w,y))

            block_detector = BlockDetector(block_roi, self.block_length)
            block_detector_vis = block_detector.visualize_block_detector(block_detector_vis)        
            block_detector_vis = cv2.rectangle(block_detector_vis, (x,y), (x+w,y+h), (255, 0, 128), 1)
        cv2.imshow("queue parser vis", block_detector_vis)
