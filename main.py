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
        
        ratio_tolerance = 0.7
    
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

def draw_poly(im, vertices, color, line_width):
    for i in range(len(vertices)):
        cv2.line(im, vertices[i], vertices[(i+1)%len(vertices)], color, line_width)

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
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
                
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        squares = []
        logo = None
        
        for i, contour in enumerate(contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
         
            #Logo detection
            if logo is None and len(approx) > 6:
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
                        
            #Square detection
            if is_square(approx):
                M = cv2.moments(approx)
                area = M['m00']
                
                if area > 100 and area < 5000:
                    center_x = int((M['m10'] / M['m00']))
                    center_y = int((M['m01'] / M['m00']))
                                                
                    square = { 'center': (center_x, center_y), 'vertices': contours_to_vertices(approx), 'area': area }
                    squares.append(square)
                    
                    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 1)
                    #cv2.putText(frame, str(i), tuple(approx[2][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
                      
        BOARD_HEIGHT = 7
        BOARD_WIDTH = 6
           
        #Remove outliers
        if len(squares) >= BOARD_HEIGHT * BOARD_WIDTH:
            new_squares = []
            
            for square_1 in squares:
                neighbors_cnt = 0
                is_inside_other = False
                for square_2 in squares:
                    if square_1 is not square_2:
                        dist_1_to_2 = euclidean_dist(square_1['center'], square_2['center'])
                        square_2_dists = []
                        for i in range(len(square_2['vertices'])):
                            dist = euclidean_dist(square_2['vertices'][i], square_2['vertices'][(i+1)%len(square_2['vertices'])])
                            square_2_dists.append(dist)
                        
                        max_square_2_dist = max(square_2_dists)
                        
                        if abs(1 - max_square_2_dist / dist_1_to_2) < 0.3:
                            neighbors_cnt += 1
                            if neighbors_cnt >= 2:
                                break
                if neighbors_cnt >= 2 and not is_inside_other:
                    new_squares.append(square_1)
            squares = new_squares
           
        for square in squares:
            draw_poly(frame, square['vertices'], (0,255,0), 2)
           
        if len(squares) >= BOARD_HEIGHT * BOARD_WIDTH and logo is not None:
            board_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for square in squares:
                cv2.fillConvexPoly(board_mask, np.array(square['vertices']), 255)
            
            kernel = np.ones((3,3), np.uint8)
            board_mask = cv2.dilate(board_mask, kernel, iterations=4)
            cv2.imshow('board_mask', board_mask)
            
            board_contours = cv2.findContours(board_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            board_contour = max(board_contours, key=cv2.contourArea)
            
            peri = cv2.arcLength(board_contour, True)
            board_approx = cv2.approxPolyDP(board_contour, 0.03 * peri, True)
            
            if len(board_approx) == 4:
                board_vertices = contours_to_vertices(board_approx)
                
                cv2.drawContours(frame, [board_approx], -1, (0, 0, 255), 2)
            
                max_dist = 0
                max_dist_center = None
                
                for i in range(len(board_vertices)):
                    first = board_vertices[i]
                    next = board_vertices[(i+1)%len(board_vertices)]
                    center = ((first[0]+next[0])//2, (first[1]+next[1])//2) 
                    dist = euclidean_dist(center, logo['center'])
                
                    if dist > max_dist:
                        max_dist = dist
                        max_dist_center = center
                
            
                cv2.circle(frame, max_dist_center, 3, (0,0,255), -1)
                cv2.line(frame, max_dist_center, logo['center'], (0,0,0), 3)
                
                angle = get_line_angle(max_dist_center, logo['center'])
                rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), angle, 1.0)
                
                board_rotated_vertices = rotate_pts(board_vertices, rotation_matrix)
                board_corners = []
                for orig, rotated in zip(board_vertices, board_rotated_vertices):
                    board_corners.append({'orig': orig, 'rotated': rotated})
                
                board_corners.sort(key=lambda i: i['rotated'][1])
                
                top_left = min(board_corners[:2], key=lambda i: i['rotated'][0])
                top_right = max(board_corners[:2], key=lambda i: i['rotated'][0])
                bottom_left = min(board_corners[2:], key=lambda i: i['rotated'][0])
                bottom_right = max(board_corners[2:], key=lambda i: i['rotated'][0])
                
                orig_vertices = (top_left['orig'], top_right['orig'], bottom_left['orig'], bottom_right['orig'])
                                
                for i, vertice in enumerate(orig_vertices):
                    cv2.putText(frame, str(i), vertice, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                dest_offset = 30
                orig_vertices = np.float32([top_left['orig'], top_right['orig'], bottom_right['orig'], bottom_left['orig']])
                dest_vertices = np.float32([(dest_offset,dest_offset), (frame.shape[1]-dest_offset,dest_offset), (frame.shape[1]-dest_offset,frame.shape[0]-dest_offset), (dest_offset,frame.shape[0]-dest_offset)])
                                    
                transformation = cv2.getPerspectiveTransform(orig_vertices, dest_vertices)
                transformed = cv2.warpPerspective(orig_frame, transformation, (orig_frame.shape[1], orig_frame.shape[0]))
                
                
                for square in squares:
                    vertices_array = np.float32([square['vertices']])
                    center_array = np.float32([[square['center']]])
                    transformed_vertices = cv2.perspectiveTransform(vertices_array, transformation)
                    transformed_center = cv2.perspectiveTransform(center_array, transformation)
                    square['vertices'] = [tuple(i) for i in np.int32(transformed_vertices[0])]
                    square['center'] = tuple(np.int32(transformed_center[0])[0])
                    
                #Get rid of squares out of board
                margin = 10
                new_squares = []
                for square in squares:
                    is_out_of_board = False
                    for vertice in square['vertices']:
                        if vertice[0] < -margin or vertice[0] >= frame.shape[1] + margin or vertice[1] < -margin or vertice[1] >= frame.shape[0] + margin:
                            is_out_of_board = True
                            break
                    if not is_out_of_board:
                        new_squares.append(square)
                squares = new_squares
                    
                squares.sort(key=lambda i: i['center'][1])
                sorted_squares = []
                
                for row_no in range(BOARD_WIDTH):
                    row_squares = squares[(row_no*BOARD_HEIGHT):((row_no+1)*BOARD_HEIGHT)]
                    row_squares.sort(key=lambda i: i['center'][0])
                    sorted_squares += row_squares
                squares = sorted_squares
                
                for square in squares:
                    top_left = (min(square['vertices'], key=lambda i: i[0])[0], min(square['vertices'], key=lambda i: i[1])[1])
                    bottom_right = (max(square['vertices'], key=lambda i: i[0])[0], max(square['vertices'], key=lambda i: i[1])[1])
                    square['bb'] = (top_left, bottom_right)
                    
                new_squares = []
                for square_1 in squares:
                    is_inside_other = False
                    for square_2 in squares:
                        if square_1 is not square_2:
                            center_1 = square_1['center']
                            (top_left_2, bottom_right_2) = square_2['bb']
                            
                            if center_1[0] > top_left_2[0] and center_1[0] < bottom_right_2[0] and center_1[1] > top_left_2[1] and center_1[1] < bottom_right_2[1]:
                                is_inside_other = True
                                break
                    if not is_inside_other:
                        new_squares.append(square_1)
                squares = new_squares
                
                for i, square in enumerate(squares):
                    cv2.rectangle(transformed, square['bb'][0], square['bb'][1], (0,255,0), 2)
                    cv2.putText(transformed, str(i), square['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    #draw_poly(transformed, square['vertices'])
                     
                     
                cv2.imshow('transformed', transformed)
                
            # top_right_center = ((top['center'][0] - right['center'[0])//2, (top['center'][1] - right['center'[1])//2)
            # right_bottom_center = ((right['center'][0] - bottom['center'[0])//2, (right['center'][1] - bottom['center'[1])//2)
            # bottom_left_center = ((bottom['center'][0] - left['center'[0])//2, (bottom['center'][1] - left['center'[1])//2)
            # left_top_center = ((left['center'][0] - top['center'[0])//2, (left['center'][1] - top['center'[1])//2)
            
            
            
            # src_vertices = np.float32([top['center'], right['center'], bottom['center'], left['center']])
            # dst_vertices = np.float32([(30,30), (frame.shape[1]-30,30), (frame.shape[1]-30,frame.shape[0]-30), (30,frame.shape[0]-30)])
            # transform = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
            
            # transformed = cv2.warpPerspective(orig_frame, transform, (orig_frame.shape[1], orig_frame.shape[0]))
            # cv2.imshow('transformed', transformed)
            
            # new_squares = []
                
            # for row_no in range(BOARD_WIDTH):
                # row_squares = squares[(row_no*BOARD_HEIGHT):((row_no+1)*BOARD_HEIGHT)]
                # row_squares.sort(key=lambda i: i['center'][0])
                
                # board += row_squares
           
        # if logo is not None and len(squares) > 2:
            # for square in squares:
                # square['dist_to_logo'] = euclidean_dist(square['center'], logo['center'])
            
            # squares.sort(key=lambda i: i['dist_to_logo'])
            
            # for square_1 in squares:
                # found = False
            
                # for square_2 in squares:
                    # if square_1 is not square_2:
                        # center_point = ((square_1['center'][0] + square_2['center'][0])//2, (square_1['center'][1] + square_2['center'][1])//2)
                        # center_to_logo_angle = get_line_angle(center_point, logo['center'])
                        # center_to_center_angle = get_line_angle(square_1['center'], square_2['center'])
                        # angle_diff = abs(center_to_logo_angle - center_to_center_angle) 
                        # centers_dist = euclidean_dist(square_1['center'], square_2['center'])
                        # side_len = euclidean_dist(square_1['vertices'][0], square_1['vertices'][1])
                        
                        # if angle_diff > 89.5 and angle_diff < 90.5 and abs(centers_dist - side_len) < 15:
                            # found = True
                            # revolutions = new_rotation_angle // 360
                            
                            # if new_rotation_angle % 360 > 270 and center_to_logo_angle < 90:
                                # new_rotation_angle = revolutions * 360 + center_to_logo_angle + 360
                            # elif new_rotation_angle % 360 < 90 and center_to_logo_angle > 270:
                                # new_rotation_angle = revolutions * 360 + center_to_logo_angle - 360
                            # else:
                                # new_rotation_angle = revolutions * 360 + center_to_logo_angle
                            
                            # cv2.line(frame, center_point, logo['center'], (0,0,255), 2)
                            # cv2.line(frame, square_1['center'], square_2['center'], (0,0,255), 2)
                            
                            
                            # #print(new_rotation_angle)
                            # cv2.line(frame, center_point, logo['center'], (255,0,0), 2)
                            # break
                # if found:
                    # break
                
        # rotation_angle = new_rotation_angle#rotation_angle * 0.8 + new_rotation_angle * 0.2
        # frame_center = (frame.shape[1]//2, frame.shape[0]//2)
        # rotation_matrix = cv2.getRotationMatrix2D(frame_center, rotation_angle, 1.0)
        # rotated = rotate_im(orig_frame, rotation_matrix)
        # orig_rotated = rotated.copy()
        
        # BOARD_HEIGHT = 7
        # BOARD_WIDTH = 6
        
        # if len(squares) == BOARD_HEIGHT * BOARD_WIDTH:
            # for square in squares:
                # square['center'] = rotate_pts([square['center']], rotation_matrix)[0]
                # square['vertices'] = rotate_pts(square['vertices'], rotation_matrix)
                # square['vertices'].sort(key=lambda i: i[1])
                # square['vertices'] = sorted(square['vertices'][:2], key=lambda i: i[0]) + sorted(square['vertices'][2:], key=lambda i: i[0], reverse=True)
                
            # squares.sort(key=lambda i: i['center'][1])
                
            # board = []
                
            # for row_no in range(BOARD_WIDTH):
                # row_squares = squares[(row_no*BOARD_HEIGHT):((row_no+1)*BOARD_HEIGHT)]
                # row_squares.sort(key=lambda i: i['center'][0])
                
                # board += row_squares
                
            # for i, square in enumerate(board):
                # #cv2.fillConvexPoly(rotated, np.array(square['vertices']), (0,255,0))
                # draw_poly(rotated, square['vertices'])
                # cv2.putText(rotated, str(i), square['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # i += 1
            
            # src_vertices = np.float32([board[0]['vertices'][0], board[6]['vertices'][1], board[41]['vertices'][2], board[35]['vertices'][3]])
            # dst_vertices = np.float32([(10,10),(rotated.shape[1]-10,10),(rotated.shape[1]-10,rotated.shape[0]-10),(10,rotated.shape[0]-10)])
            # transform = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
            
            # transformed = cv2.warpPerspective(orig_rotated, transform, (rotated.shape[1], rotated.shape[0]))
            
            # for i, square in enumerate(board):
                # pt_array = np.float32([square['vertices']])
                # transformed_vertices = cv2.perspectiveTransform(pt_array, transform)
                # square['vertices'] = [tuple(i) for i in transformed_vertices[0]]
                
                # draw_poly(transformed, square['vertices'])
                # cv2.putText(transformed, str(i), square['vertices'][3], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
            
            # cv2.imshow('transformed', transformed)
            
            # # top_left = board[0]['vertices'][0]
            # # bottom_right = board[0]['vertices'][2]
            # # print(board[0]['vertices'])
            # # roi = orig_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # # try:
                # # roi = cv2.resize(roi, (100, 100))
                # # cv2.imshow('roi', roi)
            # # except:
                # # pass
            
        
        # cv2.imshow('rotated', rotated)
        
        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)
        
        wait_time = 1 if use_camera else -1
        if ord('q') == cv2.waitKey(wait_time):
            return
    
if __name__ == '__main__':
    main()