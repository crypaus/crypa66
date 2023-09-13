import numpy as np
import cv2

from FrameGrabber.Frame import Frame


class ImageProc:
    
    @staticmethod
    def segment(frame):
        if isinstance(frame, Frame) is not True:
            frame = Frame(frame)
        
        _, background_msk = cv2.threshold(frame.hsv.v, 110, 255, cv2.THRESH_BINARY_INV)
        # _, background_msk = cv2.threshold(frame.hsv.v, 15, 255, cv2.THRESH_BINARY_INV)
        background_msk = cv2.medianBlur(background_msk, 3)

        blocks_msk = cv2.inRange(frame.hsv, np.array((0, 90, 100)), np.array((180, 255, 255)))
        blocks_msk = blocks_msk & ~background_msk

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blocks_msk = cv2.morphologyEx(blocks_msk, cv2.MORPH_CLOSE, kernel)
        # blocks_msk = cv2.medianBlur(blocks_msk, 3)

        ## New addition
        container_msk = cv2.inRange(frame.hsv, (0, 0, 0), (255, 75, 255))
        container_msk = container_msk & ~background_msk

        blocks_msk = blocks_msk & ~container_msk
        # cv2.imshow("blocks_msk", blocks_msk)
        ## ---

        container_msk = ~background_msk & ~blocks_msk
        # container_msk = cv2.medianBlur(container_msk, 3)


        _, background_msk = cv2.threshold(frame.hsv.v, 25, 255, cv2.THRESH_BINARY_INV)
        blocks_msk = cv2.inRange(frame.hsv, np.array((0, 0, 0)), np.array((180, 255, 255)))
        blocks_msk = blocks_msk & ~background_msk

        _, greedy_msk = cv2.threshold(frame.hsv.v, 25, 255, cv2.THRESH_BINARY)

        return {"bg":background_msk, "blocks":blocks_msk, "container":container_msk, "greedy":greedy_msk}

    @staticmethod
    def segment_2(frame):
        if isinstance(frame, Frame) is not True:
            frame = Frame(frame)
        
        _, background_msk = cv2.threshold(frame.hsv.v, 30, 255, cv2.THRESH_BINARY_INV)
        background_msk = cv2.medianBlur(background_msk, 3)

        blocks_msk = cv2.inRange(frame.hsv, np.array((0, 0, 0)), np.array((180, 255, 255)))
        blocks_msk = blocks_msk & ~background_msk

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blocks_msk = cv2.morphologyEx(blocks_msk, cv2.MORPH_CLOSE, kernel)

        return {"bg":background_msk, "blocks":blocks_msk}

    @staticmethod
    def detect_lines(mask):
        ## Attempt to break up horizontal lines wherever perpendicular vertical edges exist
        kernel = np.array([
            [-1],
            [-1],
            [-1],
            [ 0],
            [ 0],
            [ 1],
            [ 1]
        ])

        container_edges = cv2.filter2D(mask, -1, kernel)

        # (needed?)
        edges = cv2.Sobel(container_edges, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.normalize(edges, edges, -1, 1, cv2.NORM_MINMAX)
        _, edges = cv2.threshold(edges, -1, 255, cv2.THRESH_BINARY_INV)

        # container_edges = cv2.medianBlur(edges, 3)
        container_edges = edges.astype(np.uint8)
        # cv2.imshow("container_edges", container_edges)

        ## Attempt to eliminate stray edge pixels
        kernel = np.array([
            [0,   0,  0],
            [1,   1,  1],
            [-1, -1,  -1],
        ])

        container_edges = cv2.morphologyEx(container_edges, cv2.MORPH_HITMISS, kernel)

        ## Attempt to cull short edge/line segments
        # Create structure element for extracting horizontal lines through morphology operations
        horizontal_size = 5
        hori_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

        # Apply morphology operations
        container_edges = cv2.morphologyEx(container_edges, cv2.MORPH_OPEN, hori_struct)

        lines = cv2.HoughLinesP(
            container_edges, 3, np.pi / 2,
            threshold=30, minLineLength=60, maxLineGap=10
        )
        
        if lines is None:
            return []

        line_count = lines.shape[0]
        lines = lines.ravel().reshape(line_count, 4).tolist()
        lines = [((l[0],l[1]),(l[2],l[3])) for l in lines]
        
        return lines
