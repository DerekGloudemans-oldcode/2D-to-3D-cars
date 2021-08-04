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

from utils import fit_3D_boxes,get_avg_frame,find_vanishing_point,calc_diff,plot_vp,plot_3D_ordered
from axis_labeler import Axis_Labeler

import argparse


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
            [diff,box, direction,obj_idx,frame_idx,frame] = box_queue.get(timeout = 0)
            
        except queue.Empty:
            continue
        
        # fit box
        box_3d = fit_3D_boxes(diff,box,vp[0],vp[1],vp[2],granularity = 1e-03,e_init = 3e-01,show = False, verbose = False,obj_travel = direction)
        
        result = [box_3d,obj_idx,frame_idx,diff,vp,frame]
        results_queue.put(result)
        
        
def process_boxes(sequence,label_file,downsample = 1,SHOW = True, timeout = 20,threshold = 30):
    downsample = 2

    # load or compute average frame
    if False:
        try:
            name =  "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[0] + "_avg.png"
            avg_frame = cv2.imread(name)
            if avg_frame is None:
                raise FileNotFoundError
        except:
            avg_frame = get_avg_frame(sequence,ds = downsample).astype(np.uint8)
            name = "config/" + sequence.split("/")[-1].split(".mp4")[0].split("_")[0] + "_avg.png"
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
        show_vp = True
                        
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
        
    return

    
if __name__ == "__main__":    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("sequence",help = "p_c_000__")
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("--skip_fitting",action = "store_true")
        parser.add_argument("-framerate",help = "output video framerate", type = int,default = 10)
        parser.add_argument("-threshold",help = "diff calculation threshold", type = int,default = 30)
    
        args = parser.parse_args()
        sequence = args.sequence
        SHOW = args.show
        framerate = args.framerate
        threshold = args.threshold
        skip_fitting = args.skip_fitting
    
    except:
        skip_fitting = False
        sequence = "p3c6_00000"
        SHOW = True
        threshold = 30
        framerate = 10
        
    # define file paths
    vid_sequence = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_{}.mp4".format(sequence)
        
    #vid_sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed/{}.mp4".format(sequence)
    labels = "/home/worklab/Data/dataset_alpha/track_corrected_unique/{}_track_outputs_corrected.csv".format(sequence)
    labels_3D = "/home/worklab/Data/dataset_alpha/track_corrected_unique/{}_track_outputs_corrected_3D.csv".format(sequence)


    # if not skip_fitting:
    #     process_boxes(vid_sequence,labels,downsample = 2,SHOW = SHOW,threshold = threshold,timeout = 60)    


all_vps = {}

for p in range(1,4):
    for c in range(1,7):
        camera_id = "p{}c{}".format(p,c)
        
        name = "config/{}_axes.csv".format(camera_id)
        labels = []
        with open(name,"r") as f:
            read = csv.reader(f)
            for row in read:
                if len(row) == 5:
                    row = [int(float(item)) for item in row]
                elif len(row) > 5:
                    row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                labels.append(np.array(row))
                        
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
            
            all_vps[camera_id] = vps
