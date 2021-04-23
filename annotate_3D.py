"""
Script to add to an output 2D tracking CSV in I24 format by computing a 3D bounding box for each
vehicle fully within the frame. The resulting bounding boxes are appended to a new file

Created on Thu Apr 22 14:06:56 2021

@author: worklab
"""

import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time

from utils import fit_3D_boxes,get_avg_frame,find_vanishing_point,calc_diff,plot_vp,plot_3D
from axis_labeler import Axis_Labeler




def annotate_3d_box(box_queue,results_queue,CONTINUE,vp):
    """
    box_queue - an mp.queue with each element [diff image, 2D bbox, approx direction of travel,obj_idx,frame_idx]
    results_queue - an mp.queue to which results are written
    CONTINUE - shared mp value that indicates whether each process should continue
    vps - a list of [vp1,vp2,vp3] where each vp is (vpx,vpy) in image coordinates
    
    Repeatedly checks the queue for a box to process, and if one exists dequeues it, processes it, and writes
    resulting box (or None in case of error) to the results_queue
    """
    CONTINUE_COPY = True
    
    while CONTINUE_COPY:
        
        with CONTINUE.get_lock(): 
            CONTINUE_COPY = CONTINUE.value 
        
        try:
            [diff,box, direction,obj_idx,frame_idx] = box_queue.get(timeout = 0)
            
        except queue.Empty:
            continue
        
        # fit box
        box_3d = fit_3D_boxes(diff,box,vp[0],vp[1],vp[2],granularity = 1e-02,e_init = 3e-01,show = False, verbose = False)
        
        result = [box_3d,obj_idx,frame_idx,diff,vp]
        results_queue.put(result)
        
        
        
        
def process_boxes(sequence,label_file,downsample = 1):
    downsample = 2

    # load or compute average frame
    
    try:
        name =  "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        avg_frame = cv2.imread(name)
        if avg_frame is None:
            raise FileNotFoundError
    except:
        avg_frame = get_avg_frame(sequence,ds = downsample).astype(np.uint8)
        name = "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        cv2.imwrite(name,avg_frame)
   
    
    
    
    # get axes annotations
    try:
        name = "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_axes.csv"
        labels = []
        with open(name,"r") as f:
            read = csv.reader(f)
            for row in read:
                if len(row) == 5:
                    row = [int(float(item)) for item in row]
                elif len(row) > 5:
                    row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                labels.append(np.array(row))
        show_vp = False
                        
    except FileNotFoundError:
        labeler = Axis_Labeler(sequence,ds = downsample)
        labeler.run()
        labels = labeler.axes
        show_vp = True
    
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
        vps = [vp1,vp2,vp3]

    if show_vp:
        plot_vp(sequence,vp1 = vp1,vp2 = vp2,vp3 = vp3, ds  = downsample)
        
    # load csv file annotations into list
    box_labels = []
    with open(label_file,"r") as f:
        read = csv.reader(f)
        HEADERS = True
        for row in read:
            
            if not HEADERS:
                box_labels.append(row)
            if len(row) > 0:
                if HEADERS and row[0][0:5] == "Frame":
                    HEADERS = False # all header lines have been read at this point
        
    # load sequence with videoCapture object and get first frame
    frame_idx = 0
    cap  = cv2.VideoCapture(sequence)
    ret,frame = cap.read()
    
    # downsample first frame
    if downsample != 1:
            frame = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
    diff = calc_diff(frame,avg_frame)


    # resize average frame
    if avg_frame is None:
        avg_frame = np.zeros(frame.shape).astype(np.uint8)
    avg_frame = cv2.resize(avg_frame,(frame.shape[1],frame.shape[0]))
    
    
    # mp shared variables
    
    box_queue = mp.Queue()
    results_queue = mp.Queue()
    CONTINUE = mp.Value("i")
    with CONTINUE.get_lock(): 
        CONTINUE.value =  1
    
    # start worker processes
    pids = []
    for n in range(mp.cpu_count() - 18):
        p = mp.Process(target=annotate_3d_box, args=(box_queue,results_queue,CONTINUE,vps))
        pids.append(p)
    for p in pids:
        p.start()
    
            
    # main loop - queue and collect
    all_results = []    
    start_time = time.time()
    count = 0
    result = "None"
    errors = 0
    try:
        while len(all_results) < len(box_labels): 
            
            if count <  len(box_labels):
            
                    
                box = box_labels[count]
                # format box
                bbox = np.array(box[4:8]).astype(float) / downsample
                direction = None # TODO: the sort of thing you love to see in code -- fix later
                obj_idx = int(box[2])
                labeled_frame_idx = int(box[0])
                
                # advance current frame if necessary
                if frame_idx < labeled_frame_idx:
                    while frame_idx < labeled_frame_idx:
                        ret = cap.grab()
                        frame_idx += 1
                    ret,frame = cap.retrieve()
                    if downsample != 1:
                        frame = cv2.resize(frame,(frame.shape[1]//downsample,frame.shape[0]//downsample))
                    diff = calc_diff(frame,avg_frame)
                
                # cv2.imshow("frame",diff)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # add box to box_queue (add extra dimension so it is a list of 1 box)
                inp = [diff,[bbox],direction,obj_idx,frame_idx]
                box_queue.put(inp)
                count += 1
                
                #test 
                #fit_3D_boxes(diff,[bbox],vps[0],vps[1],vps[2],granularity = 3e-01,e_init = 1e-01,show = True, verbose = False)
                
            else:
                cap.release()
           
            
            bps = np.round(len(all_results) / (time.time() - start_time),3)
            print("\rFrame {}, {}/{} boxes queued, {}/{} 3D boxes collected ({} bps) - Errors so far: {}".format(frame_idx,count,len(box_labels),len(all_results),len(box_labels),bps,errors),end = '\r', flush = True)  
            
            # get result if any new results are ready
            try:
                result = results_queue.get(timeout = 0)
                all_results.append(result[0:2])
                
                diff = result[3]
                box_3d = result[0]
                vp = result[4]
                try:
                    fr = plot_3D(diff,box_3d[0],vp[0],vp[1],vp[2],threshold = 200)
                    cv2.imshow("3D Estimated Bboxes",fr)
                    cv2.waitKey(1)             
                except:
                    errors += 1
                
             
            except queue.Empty:
                continue
            
            
            
        
        print("\nFinished collecting processed boxes")
        cv2.destroyAllWindows()
        with CONTINUE.get_lock(): 
            CONTINUE.value =  0
        for p in pids:
            p.terminate()
            p.join()
            
            
        # organize results

        
        print("All worker processes terminated")
    
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        for p in pids:
            p.terminate()
            p.join()
        print("All worker processes terminated.")
        raise KeyboardInterrupt
    
    
    
if __name__ == "__main__":
    
    sequence = "/home/worklab/Documents/derek/2D-to-3D-cars/_data/record_p1c5_00001.mp4"
    labels = "/home/worklab/Documents/derek/2D-to-3D-cars/_data/record_p1c5_00001_track_outputs.csv"
    process_boxes(sequence,labels,downsample = 2)    
    
    


