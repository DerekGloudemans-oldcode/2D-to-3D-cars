import cv2
import cv2 as cv
import time
from os import mkdir
import os
from PIL import Image 
import numpy as np 
import csv
import sys
import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms
import time
import math
import heapq
from scipy.spatial import ConvexHull

detector_path = os.path.join(os.getcwd(),"py_ret_det_multigpu")
sys.path.insert(0,detector_path)
from py_ret_det_multigpu.retinanet.model import resnet50 

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
    avg_Frame = None
    
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
    
        print("\r On frame: {}, FPS: {:5.2f}".format(frame_num, frame_num / (time.time() - start)),end = '\r', flush = True)
    
        # get next frame
        ret, frame = cap.read()
        if frame_num > n: # early video cutoff
            cap.release()
            break
        
    avg_frame = (avg_frame/frame_num)
    return avg_frame

 
def plot_diff(video_sequence,avg_frame = None, vps = None,ds = 1):
    
    GPU_ID = 1
    detector = resnet50(num_classes=13,device_id = GPU_ID)
    cp = torch.load("./config/detector.pt")
    detector.load_state_dict(cp) 
    detector = detector.to(GPU_ID)
    detector.eval()
    detector.training = False
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_sequence)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_sequence)
    
    start = time.time()
    frame_num = 0

    
    # get first frame
    ret, frame = cap.read()
    
    if ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
    
    if avg_frame is None:
        avg_frame = np.zeros(frame.shape).astype(np.uint8)
    avg_frame = cv2.resize(avg_frame,(1920,1080))
        
    while ret: 
        frame_num += 1
            
        # get detections
        #frame = cv2.resize(frame,(1920,1080))
        im = np.array(frame)[:,:,[2,1,0]].copy()
        im = F.to_tensor(im)
        im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        
        device = torch.device("cuda:{}".format(1))
        im = im.to(device).unsqueeze(0)
        
        scores,_,boxes = detector(im)
        
        # looks like we need to do nms across all classes
        idxs = nms(boxes,scores,0.3)
        boxes = boxes[idxs]
        boxes = boxes.cpu().data.numpy()

        
        # get features
        edges = cv2.Canny(frame,250,200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        diff = np.clip(np.abs(frame.astype(int) - avg_frame.astype(int)),0,255)

        # kernel to remove small noise
        diff = cv2.blur(diff,(5,5)).astype(np.uint8)
        
        # threshold
        _,diff = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
        frame =  diff.astype(np.uint8) # + cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
       
        if vps is not None:
            fit_3D_boxes(frame, boxes, vps[0], vps[1], vps[2])
        
        #plot bboxes
        for box in boxes:
            color = (255,0,255) #colors[int(obj.cls)]
            c1 =  (int(box[0]),int(box[1]))
            c2 =  (int(box[2]),int(box[3]))
            frame = cv2.rectangle(frame,c1,c2,color,1)
            
        cv2.imshow("Difference",frame)
        cv2.waitKey(1)
        
        print("\r On frame: {}, FPS: {:5.2f}".format(frame_num, frame_num / (time.time() - start)),end = '\r', flush = True)
        torch.cuda.empty_cache()
        
        # get next frame
        ret, frame = cap.read()
        
        if ret and ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
        
        if frame_num > 200: # early video cutoff
            cap.release()
            break
       
    cv2.destroyAllWindows()


class Axis_Labeler():
    def __init__(self,directory,save_name = "temp.csv", classes = ["Along Lane (green)", "Perpendicular to Lane (blue)" , "Up (red)"],ds = 1):
        self.frame = 1
        self.ds = ds
        
        self.directory = directory
        
        self.cap = cv2.VideoCapture(directory)
        ret,frame = self.cap.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        self.frames = [frame]
        
        
        self.axes = []
        
        #self.load_annotations()
        
        self.cur_image = self.frames[0]
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.define_direction = False
        
        # classes
        self.cur_class = 0
        self.n_classes = len(classes)
        self.class_names = classes
        self.colors = (np.random.rand(self.n_classes,3))*255
        self.colors[0] = np.array([0,255,0])
        self.colors[1] = np.array([255,0,0])
        self.colors[2] = np.array([0,0,255])

        self.plot_axes()
        self.changed = False

    # def load_annotations(self):
    #     try:
    #         self.cur_frame_boxes = []
    #         name = "annotations/new/{}.csv".format(self.frames[self.frame-1].split("/")[-1].split(".")[0])
    #         with open(name,"r") as f:
    #             read = csv.reader(f)
    #             for row in read:
    #                 if len(row) == 5:
    #                     row = [int(float(item)) for item in row]
    #                 elif len(row) > 5:
    #                     row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
    #                 self.cur_frame_boxes.append(np.array(row))
                    
    #     except FileNotFoundError:
    #         self.cur_frame_boxes = []
        
    def plot_axes(self):
        self.cur_image = self.frames[self.frame-1].copy()

        last_source_idx = 0
        for box in self.axes:
                self.cur_image = cv2.line(self.cur_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),self.colors[int(box[4])],2)        
                
    def toggle_class(self):
        self.cur_class = (self.cur_class + 1) % self.n_classes
        print("Active Class: {}".format(self.class_names[self.cur_class]))
        
    def on_mouse(self,event, x, y, flags, params):
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y,self.cur_class]).astype(int)
            self.axes.append(box)
            self.new = box
            self.clicked = False
              
                 
    def next(self):
        self.clicked = False
        self.define_direction = False
        
        if self.frame == len(self.frames):
            ret,frame = self.cap.read()
            if ret:
                if self.ds != 1:
                    frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
                self.frames.append(frame)
            else:
                print("Last Frame.")    
                return
            
        self.frame += 1
        
        # load image and plot existing boxes
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        self.changed = False
                
                
            
    def quit(self):
        
        cv2.destroyAllWindows()
        self.cont = False
        self.save()
        
    def save(self):
        if len(self.axes) == 0:
            return
        sequence = self.directory.split(".")[0].split("/")[-1]
        name = sequence + ".csv"
        if os.path.exists(name): # don't overwrite
            overwrite = input("Overwrite existing file? (y/n)")
            if overwrite != "y":
                return
        
        outputs = []
        for item in self.axes:
            output = list(item)
            outputs.append(output)
        
        with open(name,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(outputs)
        print("Saved axes as file {}".format(name))
        
        name = name.split(".csv")[0] + "_axes.png"
        cv2.imwrite(name,self.cur_image)
        
        
    def clear(self):
        self.axes = []
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        
    def undo(self):
        self.clicked = False
        self.define_direction = False
        self.define_magnitude = False
        
        self.axes = self.axes[:-1]
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        
        
    def run(self):  
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           if self.new is not None:
               self.plot_axes()
                    
           self.new = None
               
           cv2.imshow("window", self.cur_image)
           title = "{} toggle class (1), switch frame (8-9), clear all (c), undo(u),   quit (q), switch frame (8-9)".format(self.class_names[self.cur_class])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord('9'):
                self.next()
           # elif key == ord('8'):
           #      self.prev()
           elif key == ord('c'):
                self.clear()
           elif key == ord("q"):
                self.quit()
           elif key == ord("1"):
                self.toggle_class()
           elif key == ord("u"):
               self.undo()
           elif key == ord("d"):
               self.remove()
     
        
def plot_vp_boxes(im,parameters,vp1,vp2,vp3):

    for idx,row in enumerate(parameters):
        # for each, we'll draw a line that ends at x = 0 and x = image width to be safe
        # y-y0 = m(x-x0)
        vp1 = (int(vp1[0]),int(vp1[1]))
        vp2 = (int(vp2[0]),int(vp2[1]))
        vp3 = (int(vp3[0]),int(vp3[1]))

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
        
    cv2.imshow("Frame",im)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    
    return im
     
def get_hulls(diff,parameters,vp1,vp2,vp3):
    start = time.time()
    
    hulls = np.zeros(diff.shape[:-1])
    
    actual_y = np.arange(0,diff.shape[0])
    actual_y = actual_y[:,np.newaxis].repeat(diff.shape[1],1)
    actual_x = np.arange(0,diff.shape[1])
    actual_x = actual_x[np.newaxis,:].repeat(diff.shape[0],0)
        
    for box in parameters:
        
        # create 6 raster maps
        # each map contains actual y-value - expected y-value based on inequality
        #expected y = m(x-x0) + y0 where x0,y0 = vp
        
        map1 = actual_y  - (box[0]*(actual_x - vp1[0]) + vp1[1])
        map2 = actual_y -  (box[1]*(actual_x - vp1[0]) + vp1[1])
        map12 = np.multiply(map1,map2) * -1         # we want the value for map12 to be negative (actual_y between two expected y)
        map12 = np.ceil(np.clip(map12,0,1))
        
        map3 = actual_y  - (box[2]*(actual_x - vp2[0]) + vp2[1])
        map4 = actual_y -  (box[3]*(actual_x - vp2[0]) + vp2[1])
        map34 = np.multiply(map3,map4) * -1         # we want the value for map12 to be negative (actual_y between two expected y)
        map34 = np.ceil(np.clip(map34,0,1))
        
        map5 = actual_y  - (box[4]*(actual_x - vp3[0]) + vp3[1])
        map6 = actual_y -  (box[5]*(actual_x - vp3[0]) + vp3[1])
        map56 = np.multiply(map5,map6) * -1         # we want the value for map12 to be negative (actual_y between two expected y)
        map56 = np.ceil(np.clip(map56,0,1))
        
        map_total = np.multiply(np.multiply(map12,map34),map56)
        
        # add to hulls (ie take union over all boxes)
        hulls += map_total
    
    # hulls = np.clip(hulls,0,1)
    # cv2.imshow("Convex Hulls for current box parameters",hulls)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #print("took {} sec".format(time.time() - start))
    return hulls

def score_hulls(hulls,diff,beta = 1):
    """
    Alpha specifies weighting of FP to FN
    """    
    diff = np.clip(np.max(diff,axis = 2),0,1)
    
    intersection = np.multiply(diff,hulls)
    
    FP = np.sum(hulls) - np.sum(intersection)
    FN = np.sum(diff) -  np.sum(intersection)
    TP = np.sum(intersection)
    
    score = TP / (TP+FP+beta*FN)
    
    return score

def init_parameters(boxes,vp1,vp2,vp3,delta = 45):
    

    # define some initial boxes (6 parameters for each box)
    # we can define each line by an angle (since we know one point, the vanishing point)
    parameters = np.zeros([len(boxes),6]) # angles for each line, for each box
    for b_idx, box in enumerate(boxes):
        # get angle to center of bbox +- a small epsilon from each vp
        bx = (box[0] + box[2]) / 2.0
        by = (box[1] + box[3]) / 2.0
        
        # find slope
        a1 = 180/math.pi*math.atan2((by-vp1[1]),(bx-vp1[0]))
        a2 = 180/math.pi*math.atan2((by-vp2[1]),(bx-vp2[0]))
        a3 = 180/math.pi*math.atan2((by-vp3[1]),(bx-vp3[0]))
        
        parameters[b_idx,:] = np.array([a1-delta,a1+delta,a2-delta,a2+delta,a3-delta,a3+delta])
        
    return parameters 
   
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
    # if direction == "greater":
    #     score = greater/(greater + less)
    # elif direction == "less":
    #     score = less/(greater + less)
        
    # return score
      
def fit_3D_boxes(diff,boxes,vp1,vp2,vp3,delta = 3,epsilon =1):
    
    # to start, we define a 3D bbox guess
    parameters = init_parameters(boxes,vp1,vp2,vp3,delta = delta)
    original_parameters = parameters.copy()
    #plot_vp_boxes(diff,parameters,vp1,vp2,vp3) 
    
    diff_clip = np.clip(np.max(diff,axis = 2),0,1)
    # make each diff object integer-unique so fitting one box is not affected by pixels in another
    for idx, box in enumerate(boxes):
        box = box.astype(int)
        diff_clip[box[1]:box[3],box[0]:box[2]] *= (idx+2)
        
    for i in range(len(parameters)):
        for j in range(len(parameters[0])):
            # determine which way to move line to contract box
            if j in [0,2,4]:
                sign = 1
            else:
                sign = -1
            
            while True:
                
                if j in [0,1]:
                    vp = vp1
                if j in [2,3]:
                    vp = vp2
                if j in [4,5]:
                    vp = vp3
                
                new_angle  = parameters[i,j] + (epsilon * sign)
                slope = math.tan(new_angle*math.pi/180)
                
                ratio = dividing_line(diff_clip,slope,vp,obj_idx = i + 2)
                
                if ratio > 0.5 :#or ratio < 1-cut_ratio:
                    parameters[i,j] = new_angle
                else:
                    sign /= 10.0
                    if np.abs(sign) < 1e-3:
                        break
    
    if True:
        slope_parameters = np.tan(parameters*np.pi/180)
        diff_new =  plot_vp_boxes(diff.copy(),slope_parameters,vp1,vp2,vp3)
    
    # parameters now define the angle of each line
    all_intersections = []
              
    # find all intersecting points and then remove the convex hull (irrelevant points)
    for i,box in enumerate(parameters):
        intersections = []
        
        for j,line_angle in enumerate(box):
            a = math.tan(line_angle *math.pi/180)
            if j in [0,1]:
                    vp = vp1
            if j in [2,3]:
                vp = vp2
            if j in [4,5]:
                vp = vp3
            
            for k,second_line_angle in enumerate(box):
                if j in [0,1] and k in [0,1] or j in [2,3] and k in [2,3] or j in [4,5] and k in [4,5] or k == j:
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
        
        # get convex hull
        intersections = np.array(intersections)
        intersections = np.unique(np.round(intersections,6),axis = 0)
        hull_indices =  np.unique(ConvexHull(intersections).simplices.reshape(-1))
        
        keepers = []
        for idx in range(len(intersections)):
            if idx not in hull_indices:
                keepers.append(intersections[idx])
        all_intersections.append(keepers)
        
    test = diff_new.copy() 
    for box in all_intersections:
      for item in box:
        px,py = item 
        test = cv2.circle(test,(int(px),int(py)),5,(100,100,100),-1)
    cv2.imshow("Frame",test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                    
    #from each intersection, we draw a line to each vanishing point in the form ax + b and store a,b
    all_box_points = []
    test = diff.copy()

    for box in all_intersections:
        all_lines = []

        for point in box:

            a = (point[1]-vp1[1])/(point[0] - vp1[0])
            b = point[1] - point[0]*a
            all_lines.append([point,vp1])
            test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp1[0]),int(vp1[1])),(100,100,100),1)
            
            a = (point[1]-vp2[1])/(point[0] - vp2[0])
            b = point[1] - point[0]*a
            all_lines.append([point,vp2])
            test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp3[0]),int(vp3[1])),(100,100,100),1)

            a = (point[1]-vp3[1])/(point[0] - vp3[0])
            b = point[1] - point[0]*a
            all_lines.append([point,vp3])
            test = cv2.line(test,(int(point[0]),int(point[1])), (int(vp3[0]),int(vp3[1])),(100,100,100),1)

            

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
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                all_points.append([px,py])
        all_points = np.array(all_points)
        all_points = np.unique(np.round(all_points,6),axis = 0)
        all_box_points.append(all_points)
        
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
            for good in good_points:
                if np.sum(np.round(np.array(point),4)) == np.sum(np.round(good,4)):
                    continue
                line = np.array([point[0],point[1],good[0],good[1]])
                dist1 = line_to_point(line,vp1)
                dist2 = line_to_point(line,vp2)
                dist3 = line_to_point(line,vp3)
                if dist1 < best1: best1 = dist1
                if dist2 < best2: best2 = dist2
                if dist3 < best3: best3 = dist3
                
            point_score = best3 + best2 + best1
            point_scores.append(point_score)
        
        ps = np.array(point_scores)
        ps_idx = np.argsort(ps)
        all_intersections[i].append(box[ps_idx[0]])
        all_intersections[i].append(box[ps_idx[1]])
        all_intersections[i].append(box[ps_idx[2]])
        all_intersections[i].append(box[ps_idx[3]])
        all_intersections[i].append(box[ps_idx[4]])
        all_intersections[i].append(box[ps_idx[5]])
        all_intersections[i].append(box[ps_idx[6]])
        all_intersections[i].append(box[ps_idx[7]])
        print(ps[ps_idx[0]],ps[ps_idx[1]])
        
    test = diff.copy() 
    for box in all_intersections:
      for item in box:
        px,py = item 
        test = cv2.circle(test,(int(px),int(py)),5,(100,100,100),-1)
    cv2.imshow("Frame",test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
        
    # for each remaining point, we need to find the slope of the line 
    # again find all intersections, and this time keep all points at which 3 lines roughly intersect
            
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
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_sequence = "/home/worklab/Desktop/test_vid.mp4"
    downsample = 2
    
    # get average frame
    try:
        name =  test_sequence.split("/")[-1].split(".mp4")[0] + "_avg.png"
        avg = cv2.imread(name)
        if avg is None:
            raise FileNotFoundError
    except:
        avg = get_avg_frame(test_sequence,ds = downsample).astype(np.uint8)
        name =  test_sequence.split("/")[-1].split(".mp4")[0] + "_avg.png"
        cv2.imwrite(name,avg)
   
    
    
    
    # get axes annotations
    try:
        name =  test_sequence.split("/")[-1].split(".mp4")[0] + ".csv"
        labels = []
        with open(name,"r") as f:
            read = csv.reader(f)
            for row in read:
                if len(row) == 5:
                    row = [int(float(item)) for item in row]
                elif len(row) > 5:
                    row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                labels.append(np.array(row))
                        
    except FileNotFoundError:
        labeler = Axis_Labeler(test_sequence,ds = downsample)
        labeler.run()
        labels = labeler.axes
    
    
    
    # get vanishing points
    if True:    
        lines1 = []
        lines2 = []
        lines3 = []
        for item in labels:
            if item[4] == 0:
                lines1.append(item)
            elif item[4] == 1:
                lines2.append(item)
            elif item[4] == 2:
                lines3.append(item)
        
        # get all axis labels for a particular axis orientation
        vp1 = find_vanishing_point(lines1)
        vp2 = find_vanishing_point(lines2)
        vp3 = find_vanishing_point(lines3)
        
        plot_vp(test_sequence,vp1 = vp1,vp2 = vp2,vp3 = vp3, ds  = downsample)
        
        
        
    
    plot_diff(test_sequence,avg_frame = avg, vps = [vp1,vp2,vp3],ds = downsample)
      
    cv2.destroyAllWindows()
    
    


# def fit_3D_boxes(diff,boxes,vp1,vp2,vp3,alpha = 0.05,beta = 10,delta = 10,epsilon = 1e-02):
#     step = 0
    
#     # to start, we define a 3D bbox guess
#     parameters = init_parameters(boxes,vp1,vp2,vp3,delta = delta)
#     #plot_vp_boxes(diff,parameters,vp1,vp2,vp3) 
    
    
                
#     best_gradient = np.inf

                
#     # compute gradient for each parameter
#     while best_gradient > 1e-05:
#         # get convex hulls
#         hulls = get_hulls(diff,parameters,vp1,vp2,vp3)
#         #plot_vp_boxes(hulls,parameters,vp1,vp2,vp3)
        
#         # score hulls based on IOU, essentially
#         base_score = score_hulls(hulls,diff,beta = beta)
#         gradient = np.zeros(parameters.shape)
        
#         for i in range(len(parameters)):
#             for j in range(len(parameters[0])):
                
                        
#                 param_copy = parameters.copy()
#                 param_copy[i,j] += epsilon
                
#                 hulls = get_hulls(diff,param_copy,vp1,vp2,vp3)
#                 score = score_hulls(hulls,diff,beta = beta)
                
#                 gradient[i,j] = (score - base_score)/epsilon
                
#         best_gradient = np.max(np.abs(gradient))
#         #best_gradient = np.abs(gradient) 
                
#         diff_hulls = np.copy(diff).astype(int) -100
#         diff_hulls[:,:,1] += (200 * hulls).astype(int)
#         diff_hulls = np.clip(diff_hulls,0,255).astype(np.uint8)
#         plot_vp_boxes(diff_hulls,parameters,vp1,vp2,vp3)
        
#         parameters = parameters + gradient * alpha

#         step += 1
#         print("Finished gradient descent step {}: best gradient {}".format(step,np.round(best_gradient,7)))
    
    