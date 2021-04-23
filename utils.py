import cv2
import time
import numpy as np 
import math
from scipy.spatial import ConvexHull

# define a global color
colors = np.round(np.random.rand(1000,3) * 255)
colors[0,:] = np.array([0,255,0])
colors[1,:] = np.array([255,0,0])
colors[2,:] = np.array([0,0,255])
colors[3,:] = np.array([0,255,255])


def get_avg_frame(video_sequence ,n = 2000,ds = 1):
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_sequence)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_sequence)
    
    start = time.time()
    frame_num = 0
    
    # get first frame
    ret, frame = cap.read()
    
    
    while ret: 
        if ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
        frame_num += 1
        frame = np.asarray(frame).astype(float)
        
        try:
            avg_frame += frame
        except:
            avg_frame = frame
    
        print("\r Averaging frame: {}, FPS: {:5.2f}".format(frame_num, frame_num / (time.time() - start)),end = '\r', flush = True)
    
        # get next frame
        ret, frame = cap.read()
        if frame_num > n: # early video cutoff
            cap.release()
            break
        
    avg_frame = (avg_frame/frame_num)
    return avg_frame


def plot_3D(frame,box,vp1,vp2,vp3,threshold = 100,color = None):
    """
    Plots 3D points as boxes, drawing only line segments that point towards vanishing points
    """
    
    if color is None:
        color = (0,255,0)
        
    for a in range(len(box)):
        ab = box[a]
        for b in range(a,len(box)):
            bb = box[b]
            line = np.array([ab[0],ab[1],bb[0],bb[1]])
            min_dist = min(line_to_point(line,vp1),
                            line_to_point(line,vp2),
                            line_to_point(line,vp3))
            if min_dist < threshold:
                
                frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
    return frame


def plot_vp_boxes(im,parameters,vp1,vp2,vp3, HIT = None):
    """
    Plots lines according to parameters, which specifies a slope and intercept per line
    """
    vp1 = (int(vp1[0]),int(vp1[1]))
    vp2 = (int(vp2[0]),int(vp2[1]))
    vp3 = (int(vp3[0]),int(vp3[1]))
        
    for idx,row in enumerate(parameters):
        # for each, we'll draw a line that ends at x = 0 and x = image width to be safe
        # y-y0 = m(x-x0)
        x = 0
        y = row[0] * (x - vp1[0]) + vp1[1]
        im = cv2.line(im,(int(x),int(y)),(vp1[0],vp1[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[0] * (x - vp1[0]) + vp1[1]
        im = cv2.line(im,(int(x),int(y)),(vp1[0],vp1[1]),colors[idx], 1)

        x = 0
        y = row[1] * (x - vp1[0]) + vp1[1]
        im = cv2.line(im,(int(x),int(y)),(vp1[0],vp1[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[1] * (x - vp1[0]) + vp1[1]
        im = cv2.line(im,(int(x),int(y)),(vp1[0],vp1[1]),colors[idx], 1)
        
        x = 0
        y = row[2] * (x - vp2[0]) + vp2[1]
        im = cv2.line(im,(int(x),int(y)),(vp2[0],vp2[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[2] * (x - vp2[0]) + vp2[1]
        im = cv2.line(im,(int(x),int(y)),(vp2[0],vp2[1]),colors[idx], 1)
        
        x = 0
        y = row[3] * (x - vp2[0]) + vp2[1]
        im = cv2.line(im,(int(x),int(y)),(vp2[0],vp2[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[3] * (x - vp2[0]) + vp2[1]
        im = cv2.line(im,(int(x),int(y)),(vp2[0],vp2[1]),colors[idx], 1)
        
        x = 0
        y = row[4] * (x - vp3[0]) + vp3[1]
        im = cv2.line(im,(int(x),int(y)),(vp3[0],vp3[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[4] * (x - vp3[0]) + vp3[1]
        im = cv2.line(im,(int(x),int(y)),(vp3[0],vp3[1]),colors[idx], 1)
        
        x = 0
        y = row[5] * (x - vp3[0]) + vp3[1]
        im = cv2.line(im,(int(x),int(y)),(vp3[0],vp3[1]),colors[idx], 1)
        
        x = im.shape[1]
        y = row[5] * (x - vp3[0]) + vp3[1]
        im = cv2.line(im,(int(x),int(y)),(vp3[0],vp3[1]),colors[idx], 1)
        
    if HIT is not None:
        [i,j] = HIT
        x = 0
        if j in [0,1]:
            vp = vp1
        if j in [2,3]:
            vp = vp2
        if j in [4,5]:
            vp = vp3
        y = parameters[i,j] * (x - vp[0]) + vp[1]
        im = cv2.line(im,(int(x),int(y)),(vp[0],vp[1]),(255,255,255), 2)
        
        x = im.shape[1]
        y = parameters[i,j] * (x - vp[0]) + vp[1]
        im = cv2.line(im,(int(x),int(y)),(vp[0],vp[1]),(255,255,255), 2)
        
    cv2.imshow("Frame",im)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    
    return im

def fit_3D_boxes(diff,boxes,vp1,vp2,vp3,e_init =1e-01,granularity = 1e-01,show = True, verbose = False):
    
    start = time.time()
    
    # to start, we define a 3D bbox guess
    parameters,eps= init_parameters(boxes,vp1,vp2,vp3,e_init = e_init)

    
    original_parameters = parameters.copy()
    #plot_vp_boxes(diff,parameters,vp1,vp2,vp3) 
    
    diff_clip = np.clip(np.max(diff,axis = 2),0,1)
    # make each diff object integer-unique so fitting one box is not affected by pixels in another
    for idx, box in enumerate(boxes):
        box = box.astype(int)
        diff_clip[box[1]:box[3],box[0]:box[2]] *= (idx+2)
      
    if verbose: print("\nInit/ preprocess took {} sec".format(time.time() - start))
    start = time.time()
    
    iterations = 0
    anomalies = []
    for i in range(len(parameters)):
        epsilons = eps[i]
        for j in range(len(parameters[0])):
            # determine which way to move line to contract box
            if j in [0,2,4]:
                sign = 1
            else:
                sign = -1
            
            iteration = 0
            while iteration <200:
                iteration += 1
                if j in [0,1]:
                    vp = vp1
                    epsilon = epsilons[0]
                if j in [2,3]:
                    vp = vp2
                    epsilon = epsilons[1]
                if j in [4,5]:
                    vp = vp3
                    epsilon = epsilons[2]
                
                new_angle  = parameters[i,j] + (epsilon * sign)
                slope = math.tan(new_angle*math.pi/180)
                
                ratio = dividing_line2(diff_clip,slope,vp,obj_idx = i + 2)
                
                if ratio > 0.5 :#or ratio < 1-cut_ratio:
                    parameters[i,j] = new_angle
                    HIT = None
                else:
                    sign /= 3.5
                    HIT = [i,j]
                    if np.abs(sign) < granularity:
                        iterations += iteration
                        break
                
                if show:
                    slope_parameters = np.tan(parameters*np.pi/180)
                    diff_new =  plot_vp_boxes(diff.copy(),slope_parameters,vp1,vp2,vp3,HIT = HIT)
            
            if iteration >= 200:
                return "Error"
    
    if verbose: print("Drawing primary lines took {} sec, {} iterations".format(time.time() - start,iterations))
    start = time.time()
    
    # parameters now define the angle of each line
    all_intersections = []
    all_intersection_vps = []
    
    # find all intersecting points and then remove the convex hull (irrelevant points)
    for i,box in enumerate(parameters):
        intersections = []
        vanishing_points = []
        
        for j,line_angle in enumerate(box):
            a = math.tan(line_angle *math.pi/180)
            if j in [0,1]:
                vp = vp1
            if j in [2,3]:
                vp = vp2
            if j in [4,5]:
                vp = vp3
                
            for k,second_line_angle in enumerate(box):
                if (j in [0,1] and k in [0,1]) or (j in [2,3] and k in [2,3]) or (j in [4,5] and k in [4,5]) or k == j:
                    continue
                
                b = math.tan(second_line_angle *math.pi/180)
                if k in [0,1]:
                    second_vp = vp1
                if k in [2,3]:
                    second_vp = vp2
                if k in [4,5]:
                    second_vp = vp3
                    
                # find y-intercept of each line
                #y-y0 = m(x-x0) where x0 = vp so y = m(0-vp[0]) + vp[1]
                    
                # mx+b form
                #y0 = ax + c
                #y1 = bx + d 
                c = -a*vp[0] + vp[1]
                d = -b*second_vp[0] + second_vp[1]
                
                px = (d-c)/(a-b)
                py = a*(d-c)/(a-b) + c
                
                intersections.append([px,py])
                vanishing_points.append([j,k])
                
        # get convex hull
        intersections = np.array(intersections)
        vanishing_points = np.array(vanishing_points)
        intersections,idxs = np.unique(np.round(intersections,6),axis = 0,return_index = True)
        vanishing_points = vanishing_points[idxs]
       
        hull_indices =  np.unique(ConvexHull(intersections).simplices.reshape(-1))
        keepers = []
        keepers_vp = []
        for idx in range(len(intersections)):
            if idx not in hull_indices:
                keepers.append(intersections[idx])
                keepers_vp.append(vanishing_points[idx])
        all_intersections.append(keepers)
        all_intersection_vps.append(keepers_vp)
    
        if verbose: print("Finding primary intersections took {} sec".format(time.time() - start))
        start = time.time()
        
    if show:
        test = diff_new.copy() 
        for box in all_intersections:
          for item in box:
            px,py = item 
            test = cv2.circle(test,(int(px),int(py)),5,(100,100,100),-1)
        cv2.imshow("Frame",test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                    
    #from each intersection, we draw a line to each vanishing point not already used to generate that point in the form ax + b and store a,b
    all_box_points = []
    test = diff.copy()

    for i,box in enumerate(all_intersections):
        all_lines = []
        polarities = []
        
        #get convex hull of intersection points to give counterclockwise ordering
        hull_order = list(ConvexHull(all_intersections[i]).vertices)
           
                
        for j, point in enumerate(box):
            point_vps = all_intersection_vps[i][j]
            
            try:
                hull_idx = hull_order.index(j)
            except ValueError:
                # point is not on covex hull
                hull_idx = 0 # not a great solution but eh
                
                
            if hull_idx % 2 == 0:
                polarity = 0
            else:
                polarity = 1
                
            color = (0,0,255) if polarity == 1 else (0,255,0)
            if 1 not in point_vps and 0 not in point_vps:
                a = (point[1]-vp1[1])/(point[0] - vp1[0])
                b = point[1] - point[0]*a
                all_lines.append([point,vp1])
                test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp1[0]),int(vp1[1])),color,1)

            elif 2 not in point_vps and 3 not in point_vps:
                a = (point[1]-vp2[1])/(point[0] - vp2[0])
                b = point[1] - point[0]*a
                all_lines.append([point,vp2])
                test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp2[0]),int(vp2[1])),color,1)

            elif 4 not in point_vps and 5 not in point_vps:
                a = (point[1]-vp3[1])/(point[0] - vp3[0])
                b = point[1] - point[0]*a
                all_lines.append([point,vp3])
                test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp3[0]),int(vp3[1])),color,1)
            
            else:
                continue
            
            polarities.append(polarity)
            

        all_lines = np.array(all_lines)
        all_lines = np.unique(np.round(all_lines,6),axis = 0)
        
        all_points = []
        for i in range(len(all_lines)):
            for j in range(i+1,len(all_lines)):
                # a,c = all_lines[i]
                # b,d = all_lines[j]
                
                # px = (d-c)/(a-b)
                # py = a*(d-c)/(a-b) + c
                
                # using formula from : https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                [[x1,y1],[x2,y2]] = all_lines[i]
                [[x3,y3],[x4,y4]] = all_lines[j]
                
                if polarities[i] != polarities[j]:
                    continue
                
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                all_points.append([px,py])
        all_points = np.array(all_points)
        all_points,counts = np.unique(np.round(all_points,6),axis = 0,return_counts = True)
        all_box_points.append(all_points)
      
    if verbose: print("Finding secondary lines and intesections took {} sec".format(time.time() - start))
    start = time.time()
    
    if show:
        cv2.imshow("Frame",test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()            
        
    
        for box in all_box_points:
            for point in box:
                px,py = point
                test = cv2.circle(test,(int(px),int(py)),5,(100,100,100),-1)       
        
        cv2.imshow("Frame",test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # given the 6 original points, find the remaining 2 points that create lines with the existing points that minimize the distance to the origins
    #TODO - do this without sorting
    for i,box in enumerate(all_box_points):
        good_points = all_intersections[i]
        point_scores = []
        for point in box:
            all_distances = []
            best1 = np.inf
            best2 = np.inf
            best3 = np.inf
            avg_dist_from_good_points = 0
            for good in good_points:
                if np.sum(np.round(np.array(point),4)) == np.sum(np.round(good,4)):
                    continue
                line = np.array([point[0],point[1],good[0],good[1]])
                dist1 = line_to_point(line,vp1)**2
                dist2 = line_to_point(line,vp2)**2
                dist3 = line_to_point(line,vp3)**2
                if dist1 < best1: best1 = dist1
                if dist2 < best2: best2 = dist2
                if dist3 < best3: best3 = dist3
                
                #avg_dist_from_good_points += ((good[1]-point[1])**2 + (good[0]-point[0])**2)
            
            point_score = best3 + best2 + best1 #+ 10*avg_dist_from_good_points/len(good_points)
            point_scores.append(point_score)
        
        ps = np.array(point_scores)
        ps_idx = np.argsort(ps)
        all_intersections[i].append(box[ps_idx[0]])
        all_intersections[i].append(box[ps_idx[1]])
        
    if verbose: print("Parsing secondary intersections took {} sec".format(time.time() - start))
    
    if show:
        test = diff.copy() 
        for box in all_intersections:
          for item in box:
            px,py = item 
            test = cv2.circle(test,(int(px),int(py)),5,(100,100,100),-1)
        cv2.imshow("Frame",test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    
    for idx in anomalies:
        all_intersections[idx] = []
    return all_intersections


def line_to_point(line,point):
    """
    Given a line defined by two points, finds the distance from that line to the third point
    line - (x0,y0,x1,y1) as floats
    point - (x,y) as floats
    Returns
    -------
    distance - float >= 0
    """
    
    numerator = np.abs((line[2]-line[0])*(line[1]-point[1]) - (line[3]-line[1])*(line[0]-point[0]))
    denominator = np.sqrt((line[2]-line[0])**2 +(line[3]-line[1])**2)
    
    return numerator / (denominator + 1e-08)
    

def find_vanishing_point(lines):
    """
    Finds best (L2 norm) vanishing point given a list of lines

    Parameters
    ----------
    lines : [(x0,y0,x1,y1), ...]

    Returns
    -------
    vp - (x,y)
    """
    
    # mx+b form
    #y0 = ax + c
    #y1 = bx + d
    
    line0 = lines[0]
    line1 = lines[1]
    a = (line0[3] - line0[1])/line0[2] - line0[0]
    b = (line1[3] - line1[1])/line1[2] - line1[0]
    c = line0[1] - a*line0[0]
    d = line1[1] - c*line1[0]
    
    # intersection
    px = (d-c)/(a-b)
    py = a*(d-c)/(a-b) + c
    best_dist = np.inf
    
    # using intersection as starting point, grid out a grid of 11 x 11 points with spacing g
    g = 1e+16
    n_pts = 31
    
    while g > 1:
        #print("Gridding at g = {}".format(g))

        # create grid centered around px,py with spacing g
        
        x_pts = np.arange(px-g*(n_pts//2),px+g*(n_pts//2),g)
        y_pts = np.arange(py-g*(n_pts//2),py+g*(n_pts//2),g)
        
        for x in x_pts:
            for y in y_pts:
                # for each point in grid, compute average distance to vanishing point
                dist = 0
                for line in lines:
                    dist += line_to_point(line,(x,y))**2
                   
                # keep best point in grid
                if dist < best_dist:
                    px = x 
                    py = y
                    best_dist = dist
                    #print("Best vp so far: ({},{}), with average distance {}".format(px,py,np.sqrt(dist/len(lines))))
    
                # regrid
        g = g / 10.0
            
    return [px,py]


def plot_vp(sequence,vp1 = None,vp2 = None, vp3 = None,mesh_width = 200, ds = 1):
    cap = cv2.VideoCapture(sequence)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(sequence)
    
    ret, frame = cap.read()
    if ds != 1:
        frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
        
    if ret:
        
        # generate endpoints
        x_pts = np.arange(0,frame.shape[1],mesh_width)
        y_pts = np.arange(0,frame.shape[0],mesh_width)
        
        edge_pts = []
        for x in x_pts:
            edge_pts.append((int(x),int(0)))
            edge_pts.append((int(x),int(frame.shape[0])))
        for y in y_pts:
            edge_pts.append((int(0),int(y)))
            edge_pts.append((int(frame.shape[1]),int(y)))  
        
        if vp1 is not None:
             for point in edge_pts:
                 frame = cv2.line(frame,(int(vp1[0]),int(vp1[1])),point,(0,255,0),1)
                 
        if vp2 is not None:
              for point in edge_pts:
                  frame = cv2.line(frame,(int(vp2[0]),int(vp2[1])),point,(255,0,0),1)
        
        if vp3 is not None:
              for point in edge_pts:
                  frame = cv2.line(frame,(int(vp3[0]),int(vp3[1])),point,(0,0,255),1)

    cv2.imshow("frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def init_parameters(boxes,vp1,vp2,vp3,delta = 45,e_init = 1e-01):
    

    # define some initial boxes (6 parameters for each box)
    # we can define each line by an angle (since we know one point, the vanishing point)
    parameters = np.zeros([len(boxes),6]) # angles for each line, for each box
    epsilons = np.zeros([len(boxes),3])
    for b_idx, box in enumerate(boxes):
        # get angle to center of bbox +- a small epsilon from each vp
        bx = (box[0] + box[2]) / 2.0
        by = (box[1] + box[3]) / 2.0
        
        # find angle
        a1 = 180/math.pi*math.atan2((by-vp1[1]),(bx-vp1[0]))
        a2 = 180/math.pi*math.atan2((by-vp2[1]),(bx-vp2[0]))
        a3 = 180/math.pi*math.atan2((by-vp3[1]),(bx-vp3[0]))
        
        # get the length of the diagonal of each box
        diag = np.sqrt((box[2]- box[0])**2 + (box[3] - box[1])**2)/2.0
        
        
        # get length to each box
        l1 = np.sqrt((vp1[1] - by)**2 + (vp1[0] - bx)**2)
        l2 = np.sqrt((vp2[1] - by)**2 + (vp2[0] - bx)**2)
        l3 = np.sqrt((vp3[1] - by)**2 + (vp3[0] - bx)**2)

        
        # using inverse tangent, find required offset angle
        delta1 = np.abs(180/math.pi*math.atan2(diag,l1))
        delta2 = np.abs(180/math.pi*math.atan2(diag,l2))
        delta3 = np.abs(180/math.pi*math.atan2(diag,l3))

        buffer = 1.5
        
        # also initialize epsilon here relative to overall box size
        parameters[b_idx,:] = np.array([a1-delta1*buffer,
                                        a1+delta1*buffer,
                                        a2-delta2*buffer,
                                        a2+delta2*buffer,
                                        a3-delta3*buffer,
                                        a3+delta3*buffer
                                        ])
        
        epsilons[b_idx,:] = np.array([delta1*e_init,delta2*e_init,delta3*e_init])
        
    return (parameters,epsilons)
   
    
def dividing_line(diff,slope,vp,obj_idx = 1):
    # get map of expected y and actual y values
    actual_y = np.arange(0,diff.shape[0])
    actual_y = actual_y[:,np.newaxis].repeat(diff.shape[1],1)
    actual_x = np.arange(0,diff.shape[1])
    actual_x = actual_x[np.newaxis,:].repeat(diff.shape[0],0)
                                             
    gt_map = actual_y - (slope*(actual_x - vp[0]) + vp[1])
    gt_map = np.ceil(np.clip(gt_map,-0.9,1)) # 1 if this value is above line, 0 otherwise
    
    targets = 1- np.clip(np.abs(diff- obj_idx),0,1)
    
    greater =  np.multiply(targets,gt_map).sum()
    less =     np.multiply(targets,1-gt_map).sum()
    
    if less == 0 or greater == 0:
        return 1
    else:
        return 0
    
def dividing_line2(diff,slope,vp,obj_idx = 1):
    """
    Instead of computing a map over all pixels, 
    instead get for each x value the y value of the line, 
    then for each x,y pair check if occupied by obj_idx
    
    We also need to do this for each y value (back-compute x) to deal with cases where slope is near infinite
    """
    linex = np.arange(diff.shape[1]).astype(int)
    liney = np.round(slope *(linex-vp[0]) + vp[1]).astype(int)
    
    liney2 = np.arange(diff.shape[0]).astype(int)
    linex2 = np.round((liney2-vp[1])/slope + vp[0]).astype(int)
    
    for idx in range(len(linex)):
        if liney[idx] < diff.shape[0] and liney[idx] >= 0:
            if diff[liney[idx],linex[idx]] == obj_idx:
                return 0
    for idx in range(len(linex2)):
        if linex2[idx] < diff.shape[1] and linex2[idx] >= 0:
            if diff[liney2[idx],linex2[idx]] == obj_idx:
                return 0
            
    
    
    return 1
    
    
    
def calc_diff(frame,avg_frame):
    diff = np.clip(np.abs(frame.astype(int) - avg_frame.astype(int)),0,255)
    
    # kernel to remove small noise
    diff = cv2.blur(diff,(5,5)).astype(np.uint8)
    
    # threshold
    _,diff = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
    diff =  diff.astype(np.uint8) # + cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    
    return diff