import cv2
import numpy as np

def rotate_im(im, rotation_matrix):
    return cv2.warpAffine(im, rotation_matrix, (im.shape[1], im.shape[0]))

def rotate_pts(pts, rotation_matrix):
    pt_array = np.array([pts])
    rotated = cv2.transform(pt_array, rotation_matrix)
    return [tuple(i) for i in rotated[0]]

def is_square(contour):
    is_really_square = False

    if len(contour) == 4:
        (x1, y1) = contour[0][0]
        (x2, y2) = contour[1][0]
        (x3, y3) = contour[2][0]
        (x4, y4) = contour[3][0]
        
        side_12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        side_23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        side_34 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
        side_41 = np.sqrt((x4-x1)**2 + (y4-y1)**2)
        diagonal_13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        diagonal_24 = np.sqrt((x4-x2)**2 + (y4-y2)**2)
        
        ratio_12_34 = side_12 / side_34
        ratio_23_41 = side_23 / side_41
        ratio_12_23 = side_12 / side_23
        ratio_13_24 = diagonal_13 / diagonal_24
        
        ratio_tolerance = 0.3
    
        is_really_square = abs(1 - ratio_12_34) < ratio_tolerance and \
                           abs(1 - ratio_23_41) < ratio_tolerance and \
                           abs(1 - ratio_12_23) < ratio_tolerance and \
                           abs(1 - ratio_13_24) < ratio_tolerance 
                           
    return is_really_square

def get_square_angles(contour):
    (x1, y1) = contour[0][0]
    (x2, y2) = contour[1][0]
    (x3, y3) = contour[2][0]
    (x4, y4) = contour[3][0]
    
    angle_1 = np.arctan2(y2-y1,x2-x1) * 180 / np.pi
    angle_2 = np.arctan2(y3-y2,x3-x2) * 180 / np.pi
    
    if angle_1 < 0:
        angle_1 = 90 - angle_1
    if angle_1 > 90:
        angle_1 = angle_1 - 90
    
    if angle_2 < 90:
        angle_2 = 180 - angle_2
    if angle_2 > 180:
        angle_2 = angle_2 - 180
    
    print(angle_1, angle_2)
    
    return (angle_1, angle_2)
    
def get_line_angle(pt1, pt2):
    return np.arctan2(pt2[1]-pt1[1],pt2[0]-pt1[0]) * 180 / np.pi + 180

def contours_to_vertices(contours):
    return [tuple(i[0]) for i in contours]
    
def euclidean_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def draw_poly(im, vertices):
    for i in range(len(vertices)):
        cv2.line(im, vertices[i], vertices[(i+1)%len(vertices)], (0,0,255), 2)

def main():

    use_camera = True
    
    if use_camera:
        cap = cv2.VideoCapture(0)
    
    rotation_angle = 0
    new_rotation_angle = 0
    
    while True:
        
        if use_camera:
            ret, frame = cap.read()
        else:
            frame = cv2.imread('1.jpg')
            frame = cv2.resize(frame, (640, 360))
        
        orig_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)        
        edges = cv2.Canny(blurred, 30, 250)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
                
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        squares = []
        logo = None
        
        for i, contour in enumerate(contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
         
            #Logo detection
            if logo is None and len(approx) > 7:
                M = cv2.moments(approx)
                area = M['m00']
                
                if area > 500:
                    center_x = int((M['m10'] / M['m00']))
                    center_y = int((M['m01'] / M['m00']))
                    vertices = contours_to_vertices(approx)
                    
                    dists = []
                    for vertice in vertices:
                        dists.append(euclidean_dist((center_x, center_y), vertice))
                    
                    is_circle = True
                    for i in dists:
                        for j in dists:
                            if abs(i - j) > 5:
                                is_circle = False
                    
                    if is_circle:
                        logo = { 'center': (center_x, center_y), 'vertices': vertices, 'radius': int(np.average(dists)) }
                        cv2.circle(frame, (center_x, center_y), 2, (255,0,0), -1)
                        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
                        cv2.circle(frame, logo['center'], logo['radius'], (0, 255, 0), 1)
            
            #Square detection
            if is_square(approx):
                M = cv2.moments(approx)
                area = M['m00']
                
                if area > 100 and area < 5000:
                    center_x = int((M['m10'] / M['m00']))
                    center_y = int((M['m01'] / M['m00']))
                                                
                    square = { 'center': (center_x, center_y), 'vertices': contours_to_vertices(approx) }
                    squares.append(square)
                    
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    #cv2.putText(frame, str(i), tuple(approx[2][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
                    
        if logo is not None and len(squares) > 2:
            for square in squares:
                square['dist_to_logo'] = euclidean_dist(square['center'], logo['center'])
            
            squares.sort(key=lambda i: i['dist_to_logo'])
            
            for square_1 in squares:
                found = False
            
                for square_2 in squares:
                    if square_1 is not square_2:
                        center_point = ((square_1['center'][0] + square_2['center'][0])//2, (square_1['center'][1] + square_2['center'][1])//2)
                        center_to_logo_angle = get_line_angle(center_point, logo['center'])
                        center_to_center_angle = get_line_angle(square_1['center'], square_2['center'])
                        angle_diff = abs(center_to_logo_angle - center_to_center_angle) 
                        centers_dist = euclidean_dist(square_1['center'], square_2['center'])
                        side_len = euclidean_dist(square_1['vertices'][0], square_1['vertices'][1])
                        
                        if angle_diff > 89.5 and angle_diff < 90.5 and abs(centers_dist - side_len) < 15:
                            found = True
                            revolutions = new_rotation_angle // 360
                            
                            if new_rotation_angle % 360 > 270 and center_to_logo_angle < 90:
                                new_rotation_angle = revolutions * 360 + center_to_logo_angle + 360
                            elif new_rotation_angle % 360 < 90 and center_to_logo_angle > 270:
                                new_rotation_angle = revolutions * 360 + center_to_logo_angle - 360
                            else:
                                new_rotation_angle = revolutions * 360 + center_to_logo_angle
                            
                            cv2.line(frame, center_point, logo['center'], (0,0,255), 2)
                            cv2.line(frame, square_1['center'], square_2['center'], (0,0,255), 2)
                            
                            
                            #print(new_rotation_angle)
                            cv2.line(frame, center_point, logo['center'], (255,0,0), 2)
                            break
                if found:
                    break
                
        rotation_angle = new_rotation_angle#rotation_angle * 0.8 + new_rotation_angle * 0.2
        frame_center = (frame.shape[1]//2, frame.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(frame_center, rotation_angle, 1.0)
        rotated = rotate_im(orig_frame, rotation_matrix)
        orig_rotated = rotated.copy()
        
        BOARD_HEIGHT = 7
        BOARD_WIDTH = 6
        
        if len(squares) == BOARD_HEIGHT * BOARD_WIDTH:
            for square in squares:
                square['center'] = rotate_pts([square['center']], rotation_matrix)[0]
                square['vertices'] = rotate_pts(square['vertices'], rotation_matrix)
                square['vertices'].sort(key=lambda i: i[1])
                square['vertices'] = sorted(square['vertices'][:2], key=lambda i: i[0]) + sorted(square['vertices'][2:], key=lambda i: i[0], reverse=True)
                
            squares.sort(key=lambda i: i['center'][1])
                
            board = []
                
            for row_no in range(BOARD_WIDTH):
                row_squares = squares[(row_no*BOARD_HEIGHT):((row_no+1)*BOARD_HEIGHT)]
                row_squares.sort(key=lambda i: i['center'][0])
                
                board += row_squares
                
            for i, square in enumerate(board):
                #cv2.fillConvexPoly(rotated, np.array(square['vertices']), (0,255,0))
                draw_poly(rotated, square['vertices'])
                cv2.putText(rotated, str(i), square['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                i += 1
            
            src_vertices = np.float32([board[0]['vertices'][0], board[6]['vertices'][1], board[41]['vertices'][2], board[35]['vertices'][3]])
            dst_vertices = np.float32([(10,10),(rotated.shape[1]-10,10),(rotated.shape[1]-10,rotated.shape[0]-10),(10,rotated.shape[0]-10)])
            transform = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
            
            transformed = cv2.warpPerspective(orig_rotated, transform, (rotated.shape[1], rotated.shape[0]))
            
            for i, square in enumerate(board):
                pt_array = np.float32([square['vertices']])
                transformed_vertices = cv2.perspectiveTransform(pt_array, transform)
                square['vertices'] = [tuple(i) for i in transformed_vertices[0]]
                
                draw_poly(transformed, square['vertices'])
                cv2.putText(transformed, str(i), square['vertices'][3], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
            
            cv2.imshow('transformed', transformed)
            
            # top_left = board[0]['vertices'][0]
            # bottom_right = board[0]['vertices'][2]
            # print(board[0]['vertices'])
            # roi = orig_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # try:
                # roi = cv2.resize(roi, (100, 100))
                # cv2.imshow('roi', roi)
            # except:
                # pass
            
        
        cv2.imshow('rotated', rotated)
        
        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)
        
        wait_time = 1 if use_camera else -1
        if ord('q') == cv2.waitKey(wait_time):
            return
    
if __name__ == '__main__':
    main()