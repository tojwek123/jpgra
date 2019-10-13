import cv2
import numpy as np

class BoardDetector:
    def __init__(self, height, width, debug=False):
        self._height = height
        self._width = width
        self._debug = debug
        
    def _sort_points_in_grid(self, vertices, width, height):
        sorted_by_y = sorted(vertices, key=lambda i: i['center'][1])
            
        sorted_in_grid = []
        for i in range(self._height):
            row = sorted_by_y[i*width:(i+1)*width]
            row.sort(key=lambda i: i['center'][0])
            sorted_in_grid += row
            
        return sorted_in_grid
        
    def _euclidean_dist(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        
    def _detect_board_rectangle(self, im):    
        blurred = cv2.GaussianBlur(im, (17, 17), 0)
        ret, bw = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(bw, 30, 250)
        
        # if self._debug:
            # cv2.imshow('blurred', blurred)
            # cv2.imshow('bw', bw)
            # cv2.imshow('edges', edges)
        
        contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        rectangles = []
        
        for i, contour in enumerate(contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            
            if len(approx) == 4:
                diagonal_1 = self._euclidean_dist(approx[0][0], approx[2][0])
                diagonal_2 = self._euclidean_dist(approx[1][0], approx[3][0])
                diagonals_ratio = diagonal_1 / diagonal_2
                
                if diagonals_ratio > 0.6 and diagonals_ratio < 1.4:                    
                    M = cv2.moments(approx)
                    
                    if M['m00'] > 1000:
                        rectangle = {'contours': approx, 'area': M['m00']}
                        rectangles.append(rectangle)
                                      
        if len(rectangles) > 0:
            board_rectangle = max(rectangles, key=lambda i: i['area'])
        else:
            board_rectangle = None
            
        return board_rectangle
        
    def _find_darkest_board_margin_angle(self, im, board_vertices):
        size_for_margin_validation = 100
                               
        orig_vertices = np.float32(board_vertices)
        dest_vertices = np.float32([(0,0), (size_for_margin_validation,0), (size_for_margin_validation,size_for_margin_validation), (0,size_for_margin_validation)])                   
        
        transformation = cv2.getPerspectiveTransform(orig_vertices, dest_vertices)
        transformed = cv2.warpPerspective(im, transformation, (size_for_margin_validation,size_for_margin_validation))
        
        if self._debug:
            cv2.imshow('transformed', transformed)
                   
        validation_margin_size = int(0.2 * size_for_margin_validation)
        ret, transformed_bw = cv2.threshold(transformed, 100, 255, cv2.THRESH_BINARY)
        
        left_margin = transformed_bw[validation_margin_size:-validation_margin_size, :validation_margin_size]
        right_margin = transformed_bw[validation_margin_size:-validation_margin_size, -validation_margin_size:]
        top_margin = transformed_bw[:validation_margin_size, validation_margin_size:-validation_margin_size]
        bottom_margin = transformed_bw[-validation_margin_size:, validation_margin_size:-validation_margin_size]
        
        # if self._debug:
            # cv2.imshow('left_margin', left_margin)
            # cv2.imshow('right_margin', right_margin)
            # cv2.imshow('top_margin', top_margin)
            # cv2.imshow('bottom_margin', bottom_margin)
        
        margins = [{'angle': 270, 'white_cnt': np.sum(left_margin == 255)},
                   {'angle': 90, 'white_cnt': np.sum(right_margin == 255)},
                   {'angle': 0, 'white_cnt': np.sum(top_margin == 255)},
                   {'angle': 180, 'white_cnt': np.sum(bottom_margin == 255)}]
                   
        darkest_margin = min(margins, key=lambda i: i['white_cnt'])
        darkest_margin_threshold = 30
        
        if darkest_margin['white_cnt'] < darkest_margin_threshold:
            darkest_margin_angle = darkest_margin['angle']
        else:
            darkest_margin_angle = None
            
        return darkest_margin_angle
        
    def _normalize_board(self, im, board_vertices, darkest_margin_angle):
        if darkest_margin_angle == 90:
            board_vertices = board_vertices[1:] + [board_vertices[0]]
        elif darkest_margin_angle == 180:
            board_vertices = board_vertices[2:] + board_vertices[:2]
        elif darkest_margin_angle == 270:
            board_vertices = board_vertices[3:] + board_vertices[:3]
        
        normalized_board_height = 640
        normalized_board_width = 480
        
        orig_vertices = np.float32(board_vertices)
        dest_vertices = np.float32([(0,0), (normalized_board_width,0), (normalized_board_width,normalized_board_height), (0,normalized_board_height)])                   
                
        normalized_transformation = cv2.getPerspectiveTransform(orig_vertices, dest_vertices)
        normalized = cv2.warpPerspective(im, normalized_transformation, (normalized_board_width,normalized_board_height))
        
        return normalized
        
    def _find_board_fields(self, normalized):
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        if self._debug:
            cv2.imshow('fields_blurred', blurred)
        
        ret, normalized_bw = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
                            
        kernel = np.ones((3,3), np.uint8)
        normalized_bw = cv2.erode(normalized_bw, kernel, iterations=4)
        #normalized_bw = cv2.dilate(normalized_bw, kernel, iterations=3)
        
        if self._debug:
            cv2.imshow('normalized_bw', normalized_bw)
        
        board_fields = []
        contours = cv2.findContours(normalized_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        for i, contour in enumerate(contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
         
         
            if len(approx) == 4:
                side_lengths = [self._euclidean_dist(approx[i][0], approx[(i+1)%4][0]) for i in range(4)]
                
                side_0_1_ratio = side_lengths[0] / side_lengths[1]
                side_0_2_ratio = side_lengths[0] / side_lengths[2]
                side_1_3_ratio = side_lengths[1] / side_lengths[3]
                
                if side_0_1_ratio > 0.6 and side_0_1_ratio < 1.4 and \
                   side_0_2_ratio > 0.6 and side_0_2_ratio < 1.4 and \
                   side_1_3_ratio > 0.6 and side_1_3_ratio < 1.4:
                    
                    M = cv2.moments(approx)
                    
                    if M['m00'] > 100 and M['m00'] < 6000:
                        center_x = int((M['m10'] / M['m00']))
                        center_y = int((M['m01'] / M['m00']))
                        
                        vertices = [tuple(i[0]) for i in approx]
                        sorted_by_x = sorted(vertices, key=lambda i: i[0])
                        sorted_by_y = sorted(vertices, key=lambda i: i[1])
                        
                        top_left = (sorted_by_x[1][0], sorted_by_y[1][1])
                        bottom_right = (sorted_by_x[2][0], sorted_by_y[2][1])
                        bb = {'top_left': top_left, 'bottom_right': bottom_right}
                        
                        board_fields.append({'contours': approx, 'center': (center_x, center_y), 'bb': bb})
                  
        if len(board_fields) == self._width * self._height:
            board_fields = self._sort_points_in_grid(board_fields, self._width, self._height)
        else:
            board_fields = None
            
        return board_fields
        
    def detect(self, im):
        detection = None
    
        if self._debug:
            debug_im = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)
    
        board_rectangle = self._detect_board_rectangle(im)
        
        if board_rectangle is not None:
            
            if self._debug:
                cv2.drawContours(debug_im, [board_rectangle['contours']], -1, (0, 0, 255), 1)
                   
            board_vertices = [tuple(i[0]) for i in board_rectangle['contours']]
            board_vertices.sort(key=lambda i: i[1])
        
            top_left = min(board_vertices[:2], key=lambda i: i[0])
            top_right = max(board_vertices[:2], key=lambda i: i[0])
            bottom_left = min(board_vertices[2:], key=lambda i: i[0])
            bottom_right = max(board_vertices[2:], key=lambda i: i[0])
            board_vertices = [top_left, top_right, bottom_right, bottom_left]
            
            darkest_margin_angle = self._find_darkest_board_margin_angle(im, board_vertices)
            
            if darkest_margin_angle is not None:
                normalized = self._normalize_board(im, board_vertices, darkest_margin_angle)
                
                if self._debug:
                    cv2.imshow('normalized', normalized)
                
                board_fields = self._find_board_fields(normalized)
                
                if board_fields is not None:
                    detection = {'normalized': normalized, 'board_fields': board_fields}
                 
        if self._debug:
            cv2.imshow('debug_im', debug_im)
        
        return detection

def main():
    board_detector = BoardDetector(7, 6, True)

    for i in range(1,9):
        im = cv2.imread('new_board/{}.png'.format(i))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imshow('im', gray)
        
        detection = board_detector.detect(gray)
        
        if detection is not None:
            print('{} detected'.format(i))
            normalized_color = cv2.cvtColor(detection['normalized'].copy(), cv2.COLOR_GRAY2BGR)
            
            for i, field in enumerate(detection['board_fields']):
                cv2.rectangle(normalized_color, field['bb']['top_left'], field['bb']['bottom_right'], (0, 255, 0), 1)
                cv2.putText(normalized_color, str(i), field['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    
            cv2.imshow('nomalized', normalized_color)
        else:
            print('{} not detected'.format(i))
       
        cv2.waitKey()

if __name__ == '__main__':
    main()