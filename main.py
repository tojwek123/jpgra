import cv2
import numpy as np

def imrotate(im, angle_deg):
    pt = (im.shape[1]/2, im.shape[0]/2)
    r = cv2.getRotationMatrix2D(pt, angle_deg, 1.0)
    return cv2.warpAffine(im, r, (im.shape[1], im.shape[0]))

def main():

    use_camera = False
    
    if use_camera:
        cap = cv2.VideoCapture(0)
    
    while True:
        
        if use_camera:
            ret, frame = cap.read()
        else:
            frame = cv2.imread('1.jpg')
            frame = cv2.resize(frame, (640, 360))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)        
        edges = cv2.Canny(blurred, 10, 100)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        
        # blurred = cv2.GaussianBlur(gray, (3, 3), 11)     
        # t = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow('t', t)
        # circles = cv2.HoughCircles(t, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=5,maxRadius=250)

        # if circles is not None:
            # for circle in circles[0,:]:
               # center = (circle[0], circle[1])
               # radius = circle[2]
               # print('circle', circle)
               
               # cv2.circle(frame, center, radius, (0,0,255),2)
        
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        for i, contour in enumerate(contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
         
            #Four vertices means it's a rectangle
            if len(approx) > 7:
                M = cv2.moments(contour)
                area = M['m00']
                
                if area > 500:
                    center_x = int((M['m10'] / M['m00']))
                    center_y = int((M['m01'] / M['m00']))
                    
                    is_circle = True
                    x, y = approx[0][0]
                    first_distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
                    
                    for vertice in approx[1:]:
                        x, y = vertice[0]
                        distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
                        
                        # print(first_distance, distance)
                        
                        if abs(first_distance - distance) > 5:
                            is_circle = False
                            break
                       
                    if is_circle:
                        cv2.circle(frame, (center_x, center_y), 2, (255,0,0), -1)
                        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 1)
                
            
            if len(approx) == 4:
                #Check if it's square
                (x1, y1) = approx[0][0]
                (x2, y2) = approx[1][0]
                (x3, y3) = approx[2][0]
                (x4, y4) = approx[3][0]
                
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
                
                angles_1 = []
                angles_2 = []
                
                if abs(1 - ratio_12_34) < ratio_tolerance and \
                   abs(1 - ratio_23_41) < ratio_tolerance and \
                   abs(1 - ratio_12_23) < ratio_tolerance and \
                   abs(1 - ratio_13_24) < ratio_tolerance and \
                   side_12 > 20:
                    # angle_1 = np.arctan2(y2-y1,x2-x1) * 180 / np.pi + 180
                    # angle_2 = np.arctan2(y3-y2,x3-x2) * 180 / np.pi + 180
                    
                    # if angle_1 > angle_2:
                        # angle_1, angle_2 = angle_2, angle_1
                    
                    # angles_1.append(angle_1)
                    # angles_2.append(angle_2)
                    
                    # if angle_1_deg < 0:
                        # angle_1_deg += 90
                    # if angle_2_deg < 90:
                        # angle_2_deg += 90
                   
                    
                
                    screenCnt = approx
                    cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)
                    #cv2.line(frame, tuple(approx[0][0]), tuple(approx[1][0]), (255,0,0))
                    cv2.putText(frame, str(i), tuple(approx[2][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        
                # if len(angles_1) > 0:
                    # angle_1 = sum(angles_1) / len(angles_1)
                    # angle_2 = sum(angles_2) / len(angles_2)
                    
                    # print(angle_1, angle_2)
                    
                    # rotated = imrotate(frame, angle_1)
                    # cv2.imshow('rotated', rotated)
                
        # # cv2.imshow('dilated', dilated)
        
        # lines = cv2.HoughLines(edges,1,np.pi/180,100)
                
        # thetas_deg = []
        
        # if lines is not None:
            # for line in lines:
                # rho,theta = line[0]
                
                # thetas_deg.append(theta * 180 / np.pi)
                
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a*rho
                # y0 = b*rho
                # x1 = int(x0 + 1000*(-b))
                # y1 = int(y0 + 1000*(a))
                # x2 = int(x0 - 1000*(-b))
                # y2 = int(y0 - 1000*(a))
                # cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
                    
            # thetas_deg.sort()
            # horizontal_angles_deg = []
            
            # print(thetas_deg)
            
            # #Find most common angle
            # for theta_deg in thetas_deg:
                # if theta_deg < (180 + 60) and theta_deg > (180 - 60):
                    # horizontal_angles_deg.append(theta_deg)
                    
            # if len(horizontal_angles_deg) > 0:
                # horizontal_rotation_deg = sum(horizontal_angles_deg) / len(horizontal_angles_deg) - 180
                # print(horizontal_rotation_deg)
            
                    
            # for line_1 in lines:
                # for line_2 in lines:
                    # if (line_1 != line_2).all():
                        # rho_1, theta_1 = line_1[0]
                        # rho_2, theta_2 = line_2[0]
                        
                        # theta_1_deg = theta_1 * 180 / np.pi
                        # theta_2_deg = theta_2 * 180 / np.pi
                        # theta_diff_deg = theta_1_deg - theta_2_deg
                        
                        # if theta_diff_deg > 85 and theta_diff_deg < 95:
                            # y = (rho_2*np.cos(theta_1) - rho_1*np.cos(theta_2)) / (np.sin(theta_2)*np.cos(theta_1) - np.sin(theta_1)*np.cos(theta_2))
                            # x = (rho_1 - y*np.sin(theta_1)) / np.cos(theta_1)
                            
                            # cv2.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)
                    
        
        # # rotated = imrotate(frame, horizontal_rotation_deg)          
        
        # # cv2.imshow('thresh', thresh)
        cv2.imshow('frame', frame)
        # cv2.imshow('rotated', rotated)
        # cv2.imshow('blurred', blurred)
        cv2.imshow('edges', edges)
        
        wait_time = 1 if use_camera else -1
        if ord('q') == cv2.waitKey(wait_time):
            return
    
if __name__ == '__main__':
    main()