import cv2
import numpy as np

def imrotate(im, angle_deg):
    pt = (im.shape[1]/2, im.shape[0]/2)
    r = cv2.getRotationMatrix2D(pt, angle_deg, 1.0)
    return cv2.warpAffine(im, r, (im.shape[1], im.shape[0]))

def is_circle(contour):
    pass

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
        
        ratio_tolerance = 0.1
    
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
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0)        
        edges = cv2.Canny(gray, 30, 250)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
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
                        
                        if angle_diff > 88 and angle_diff < 92 and abs(centers_dist - side_len) < 15:
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
                            
                            
                            print(new_rotation_angle)
                            cv2.line(frame, center_point, logo['center'], (255,0,0), 2)
                            break
                if found:
                    break
        
        rotation_angle = rotation_angle * 0.8 + new_rotation_angle * 0.2
        rotated = imrotate(orig_frame, rotation_angle - 90)     
        cv2.imshow('rotated', rotated)
        
        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)
        
        wait_time = 1 if use_camera else -1
        if ord('q') == cv2.waitKey(wait_time):
            return
    
if __name__ == '__main__':
    main()