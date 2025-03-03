from flask import Flask
from flask import render_template, Response
from flask import request
import pandas as pd
import os
import numpy as np
import pandas as pd
import json
import csv
import cv2  # OpenCV library for computer vision tasks
import skimage  # Library for image processing
from PIL import Image  # Python Imaging Library for image manipulation
from ultralytics import YOLO  # YOLOv5 model for object detection
from sklearn.metrics import mean_squared_error  # Function to calculate mean squared error

import json  # Library for JSON data manipulation
import yaml  # Library for YAML data manipulation
import time  # Library for time-related functions


import matplotlib.pyplot as plt  # Module for plotting graphs in Python 
import numpy as np  # Module for numerical computation in Python
import pandas as pd  # Module for data manipulation and analysis in Python


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000

#################################################
# Deployment of third solution - Video Analysis #
#################################################

def match_players_positions(prev, curr):
    idx_processed = []
    curr_copy = []
    prev_copy = []
    for idx_prev_pos in range(len(prev)):
        # Calculate the minimum distance between the current position of prev and all positions of curr
        min_dist = float('inf')
        min_idx_curr_pos = -1
        for idx_curr_pos in range(len(curr)):
            if idx_curr_pos not in idx_processed:
                # Calculate the distance between the current position of prev and the current position of curr
                dist = np.linalg.norm(prev[idx_prev_pos] - curr[idx_curr_pos])
                # Update the minimum distance and the index of the current position of curr if necessary
                if dist < min_dist:
                    min_dist = dist
                    min_idx_curr_pos = idx_curr_pos
        # Swap corresponding positions in prev and curr
        if min_idx_curr_pos != -1:
            idx_processed.append(min_idx_curr_pos)
            #print(idx_processed)
            prev_copy.append(prev[idx_prev_pos])
            curr_copy.append(curr[min_idx_curr_pos])

    prev_copy = np.array(prev_copy)
    curr_copy = np.array(curr_copy)
    for id in range(len(prev)):
        if (prev_copy[id] == np.array([0,0])).all() and (curr_copy[id] != np.array([0,0])).all():
            prev_copy[id] = curr_copy[id]
        elif (prev_copy[id] != np.array([0,0])).all() and (curr_copy[id] == np.array([0,0])).all():
            curr_copy[id] = prev[id]
                
    return prev_copy, curr_copy

def is_inside_field(ball_pos):
    # Check if the ball position is inside the defined field
    xmin, xmax, ymin, ymax = 15, 291, 15, 521
    return xmin <= ball_pos[0] <= xmax and ymin <= ball_pos[1] <= ymax

def player_pos_near_ball(ball_pos, players_pos):
    # Find the index of the player nearest to the ball
    min_dist = float('inf')  # Initialize the minimum distance
    idx_min_pos = -1  # Initialize the index of the nearest player
    for idx, pos in enumerate(players_pos):
        # Calculate the Euclidean distance between the player's position and the ball
        dist = np.linalg.norm(pos - ball_pos)
        # Update the minimum distance and index of the nearest player if necessary
        if dist < min_dist:
            min_dist = dist
            idx_min_pos = idx
    return idx_min_pos
    
def calcul_possession_match(ball_pos, players_pos, players_teams_list, possession_count_dict, possession_param_dict):
    # Check if the ball is inside the field
    # Check if the ball is detected and inside the field
    if ball_pos is not None:
        if is_inside_field(ball_pos):    
            # Identify the team of the player nearest to the ball
            nearest_player_index = player_pos_near_ball(ball_pos, players_pos)
            nearest_player_team = players_teams_list[nearest_player_index]
        else:
            nearest_player_team = possession_param_dict["last_possession_team"]
    else:
        nearest_player_team = possession_param_dict["last_possession_team"]
    
    # Update the possession counters based on the team possessing the ball
    if nearest_player_team == 0:
        possession_count_dict["team0_count"] += 1  
        possession_param_dict["team_0_possession_frames"] += 1
        possession_param_dict["last_possession_team"] = 0
        # Check if team 0 possesses the ball for a sufficient number of consecutive frames
        if possession_param_dict["team_0_possession_frames"] >= possession_param_dict["consecutive_frames"]:
            # Check if team 1 possesses the ball for fewer consecutive frames
            if possession_param_dict["team_1_possession_frames"] < possession_param_dict["consecutive_frames"]:
                # Subtract frames from team 1's possession counter and add them to team 0's counter
                possession_count_dict["team1_count"] -= possession_param_dict["team_1_possession_frames"]
                possession_count_dict["team0_count"] += possession_param_dict["team_1_possession_frames"]
            possession_param_dict["team_1_possession_frames"] = 0
    if nearest_player_team == 1:
        possession_count_dict["team1_count"] += 1  
        possession_param_dict["team_1_possession_frames"] += 1
        possession_param_dict["last_possession_team"] = 1
        # Check if team 1 possesses the ball for a sufficient number of consecutive frames
        if possession_param_dict["team_1_possession_frames"] >= possession_param_dict["consecutive_frames"]:
            # Check if team 0 possesses the ball for fewer consecutive frames
            if possession_param_dict["team_0_possession_frames"] < possession_param_dict["consecutive_frames"]:
                # Subtract frames from team 0's possession counter and add them to team 1's counter
                possession_count_dict["team0_count"] -= possession_param_dict["team_0_possession_frames"]
                possession_count_dict["team1_count"] += possession_param_dict["team_0_possession_frames"]
            possession_param_dict["team_0_possession_frames"] = 0

    return possession_count_dict, possession_param_dict

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def dataframe_to_image(df):
    fig, ax = plt.subplots(figsize=(6, 8))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Convert plot to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def generate_frames(data):
    print(data)
    # Get tactical map keypoints positions dictionary
    json_path = "./Video_Analysis/data/pitch map labels position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)
    
    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "./Video_Analysis/data/config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']
    
    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "./Video_Analysis/data/config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    # Set video path
    video_path = data["video_path"]
    
    # Read tactical map image
    tac_map = cv2.imread('./Video_Analysis/data/tactical map.jpg')
    
    # Define team colors (based on chosen video)
    nbr_team_colors = 2
    colors_dic = {
        data["team_home_name"]:[data["team_home_color"], data["team_home_gk_color"]], # home colors (Players kit color, GK kit color)
        data["team_away_name"]:[data["team_away_color"], data["team_away_gk_color"]] # away colors (Players kit color, GK kit color)
    }
    # Load the YOLOv8 players detection model
    model_players = YOLO("./Video_Analysis/models/Yolo8L Players/weights/best.pt")
    
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("./Video_Analysis/models/Yolo8M Field Keypoints/weights/best.pt")
    
    colors_list = colors_dic[data["team_home_name"]]+colors_dic[data["team_away_name"]] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    
    ## Open video file
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    time_between_frames = 1 / fps

    # Initialize frame counter
    frame_nbr = 0
    
    # Set keypoints average displacement tolerance level (in pixels) [set to -1 to always update homography matrix]
    keypoints_displacement_mean_tol = data["keypoints_rmse_tolerance"]
    
    # Set confidence thresholds for players and field keypoints detections
    player_model_conf_thresh = data["detection_confidence_threshold"]
    keypoints_model_conf_thresh = data["keypoints_confidence_threshold"]
    
    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history
    ball_track_history = {'src':[],
                          'dst':[]
    }
    
    # Count consecutive frames with no ball detected
    nbr_frames_no_ball = 0
    # Threshold for number of frames with no ball to reset ball track (frames)
    nbr_frames_no_ball_thresh = data["ball_track_reset_threshold"]
    # Distance threshold for ball tracking (pixels)
    ball_track_dist_thresh = data["ball_track_distance_threshold"]
    # Maximum ball track length (detections)
    max_track_length = data["max_ball_track_length"]

    prev =[]
    # Dictionary to store all players total displacement and speed
    dict_displacement_speed ={
        "players_id" : [f"player{i}" for i in range(22)],
        "displacement (m)" : [0 for i in range(22)],
        "speed (m/s)" : [0 for i in range(22)]
    }
    
    # Dictionary containing possession parameters
    possession_param_dict = {"consecutive_frames": 5,               # Number of consecutive frames to count possession
                             "team_0_possession_frames": 0,        # Number of frames where team 0 possesses the ball
                             "team_1_possession_frames": 0,        # Number of frames where team 1 possesses the ball
                             "last_possession_team": -1}            # Index of the team possessing the ball previously (previous frame)
    
    # Dictionary to count the number of frames each team possesses the ball
    possession_count_dict = {"team0_count": 0, "team1_count": 0}
    
    # Loop through the video frames
    while cap.isOpened():
    
        # Update frame counter
        frame_nbr += 1
    
        # Read a frame from the video
        success, frame = cap.read()
    
        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()
    
        # Reset ball tracks
        if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
                ball_track_history['dst'] = []
                ball_track_history['src'] = []
    
        # Process the frame if it was successfuly read
        if success:
            
            #################### Part 1 ####################
            # Object Detection & Coordiante Transofrmation #
            ################################################
    
            # Run YOLOv8 players inference on the frame
            results_players = model_players(frame, conf=player_model_conf_thresh)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=keypoints_model_conf_thresh)
    
            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          # Detected players, referees and ball (x,y,x,y) bounding boxes
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        # Detected players, referees and ball (x,y,w,h) bounding boxes    
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        # Detected field keypoints (x,y,w,h) bounding boxes
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                        # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list
    
            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]
    
            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])
    
            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])
    
    
            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   # Get labels indexes of common detected keypoints from previous frame
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        # Get labels indexes of common detected keypoints from current frame
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     # Get labels coordiantes of common detected keypoints from previous frame
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          # Get labels coordiantes of common detected keypoints from current frame
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                        update_homography = coor_error > keypoints_displacement_mean_tol                 # Check if error surpassed the predefined tolerance level
                    else:
                        update_homography = True                                                         
                else:
                    update_homography = True
    
                if  update_homography:
                    h, mask = cv2.findHomography(detected_labels_src_pts,                   # Calculate homography matrix
                                                  detected_labels_dst_pts)                  
                
                detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame
    
                bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected players (label 0)
                bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)
    
                # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None
    
                # Transform players coordinates from frame plane to tactical map plane using the calculated Homography matrix
                pred_dst_pts = []                                                           # Initialize players tactical map coordiantes list
                for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                    dest_point = np.matmul(h, np.transpose(pt))                              # Apply homography transofrmation
                    dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)
    
                #print(pred_dst_pts)
                curr = pred_dst_pts.copy()
                curr[:, 0] = (curr[:, 0] - 15)*90/276 # 90 est la dimension (largeur) du terrain, 314-38 pour soustraire la marge
                curr[:, 1] = (curr[:, 1] -15)*120/506 # 120 est la dimension (longueur) du terrain, 546-40 pour soustraire la marge
                
                # Add null positions [0, 0] if the number of rows is less than 22 (the total number of player in the Match)
                while len(curr) < 22:
                    curr = np.vstack([curr, [0, 0]])
                    
                # Afficher le résultat des positions apres conversion de dim tactical map à dim réel de terrain
                #print(pred_dst_pts)
                
                # Calculate player total displacement and average speed
                if len(prev) != 0:
                    prev, curr = match_players_positions(prev, curr)
                    for i in range(22):
                        # Calculate the distance between current and previous player positions
                        displacement = np.linalg.norm(prev[i] - curr[i])
                        if displacement >0.4:
                            displacement = 0.2
                        dict_displacement_speed["displacement (m)"][i] += displacement
                # Update previous player positions
                prev = curr.copy()
    
                # Calculate Average speed for each player
                for i in range(22):
                    dict_displacement_speed["displacement (m)"][i] = round(dict_displacement_speed["displacement (m)"][i], 2)
                    dict_displacement_speed["speed (m/s)"][i] = round(dict_displacement_speed["displacement (m)"][i]/(time_between_frames*frame_nbr), 2)
                # Créer un DataFrame à partir du dictionnaire dict_displacement_speed
                df = pd.DataFrame(dict_displacement_speed)
                # Afficher le DataFrame
                dataframe_img = dataframe_to_image(df)
    
                # Transform ball coordinates from frame plane to tactical map plane using the calculated Homography matrix
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(h, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)
    
                    # Update track ball position history
                    if data["show_ball_tracks"]:
                        if len(ball_track_history['src'])>0 :
                            if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh:
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                # Remove oldest tracked ball postion if track exceedes threshold        
                if len(ball_track_history) > max_track_length:
                        ball_track_history['src'].pop(0)
                        ball_track_history['dst'].pop(0)
    
            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                      # Convert frame to RGB
            obj_palette_list = []                                                                   # Initialize players color palette list
            palette_interval = (0,data["palette_colors"])                                            # Color interval to extract from dominant colors palette (1rd to 5th color)
            annotated_frame = frame                                                                 # Create annotated frame 
    
            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                        
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                    palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                    palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                          by = 'cnt', ascending = False).iloc[
                                              palette_interval[0]:palette_interval[1],:]
                    palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                    
                    # Update detected players color palette list
                    obj_palette_list.append(palette)
            
            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            # Loop over detected players extracted color palettes
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
                # Loop over colors in palette
                for color in palette_lab:
                    distance_list = []
                    # Loop over predefined list of teams colors
                    for c in color_list_lab:
                        #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                        distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                        distance_list.append(distance)                                              # Update distance list for current color
                    palette_distance.append(distance_list)                                          # Update distance list for current palette
                players_distance_features.append(palette_distance)                                  # Update distance features list
    
            ## Predict detected players teams based on distance features
            players_teams_list = []
            # Loop over players distance features
            for distance_feats in players_distance_features:
                vote_list=[]
                # Loop over distances for each color 
                for dist_list in distance_feats:
                    team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
                    vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
                players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting
    
            #Call the match possession calculate function
            if detected_ball_src_pos is not None:
                possession_count_dict, possession_param_dict = calcul_possession_match(detected_ball_dst_pos[:-1], pred_dst_pts, players_teams_list, possession_count_dict, possession_param_dict)
                
            total_frame_ball_detected = possession_count_dict["team0_count"]+possession_count_dict["team1_count"]
            if total_frame_ball_detected ==0:
                total_frame_ball_detected = 1
            team1 = round(possession_count_dict["team0_count"]*100/total_frame_ball_detected)
            team2 = round(possession_count_dict["team1_count"]*100/total_frame_ball_detected)
            print(possession_count_dict, team1, team2)
    
            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
    
            ball_color_bgr = (0,0,255)                                                                          # Color (GBR) for ball annotation on tactical map
            j=0                                                                                                 # Initializing counter of detected players
            palette_box_size = 10                                                                               # Set color box size in pixels (for display)
            
    
            # Loop over all detected object by players detection model
            for i in range(bboxes_p.shape[0]):
                conf = confs_p[i]                                                                               # Get confidence of current detected object
                if labels_p[i]==0:                                                                              # Display annotation for detected players (label 0)
                    
                    # Display extracted color palette for each detected player
                    palette = obj_palette_list[j]                                                               # Get color palette of the detected player
                    
                    if data["show_color_palettes"]:
                        for k, c in enumerate(palette):
                            c_bgr = c[::-1]                                                                         # Convert color to BGR
                            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,2])+3,                 # Add color palette annotation on frame
                                                                    int(bboxes_p[i,1])+k*palette_box_size),
                                                                    (int(bboxes_p[i,2])+palette_box_size,
                                                                    int(bboxes_p[i,1])+(palette_box_size)*(k+1)),
                                                                    c_bgr, -1)
        
                    team_name = list(colors_dic.keys())[players_teams_list[j]]                                  # Get detected player team prediction
                    color_rgb = colors_dic[team_name][0]                                                        # Get detected player team color
                    color_bgr = color_rgb[::-1]                                                                 # Convert color to bgr
    
                    if data["show_player_detection"]:
                        annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add bbox annotations with team colors
                                                        (int(bboxes_p[i,2]), int(bboxes_p[i,3])), color_bgr, 1)
                        
                        cv2.putText(annotated_frame, team_name + f" {conf:.2f}",                                    # Add team name annotations
                                    (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_bgr, 2)
                    
                    # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
                    if len(detected_labels_src_pts)>3:
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                              radius=6, color=(0, 0, 0), thickness=-1)
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                              radius=5, color=color_bgr, thickness=-1)
    
                    j+=1                                                                                        # Update players counter
                else:                                                                                           # Display annotation for otehr detections (label 1, 2)
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add white colored bbox annotations
                                                     (int(bboxes_p[i,2]), int(bboxes_p[i,3])), (255,255,255), 1)
                    cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",                      # Add white colored label text annotations
                                (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255,255,255), 2)
    
                    # Add tactical map ball postion annotation if detected
                    if detected_ball_src_pos is not None:
                        tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]), 
                                                       int(detected_ball_dst_pos[1])), radius=5, 
                                                       color=ball_color_bgr, thickness=3)
            
            if data["show_keypoints_detections"]:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i,0]), int(bboxes_k[i,1])),  # Add bbox annotations with team colors
                                                (int(bboxes_k[i,2]), int(bboxes_k[i,3])), (0,0,0), 1)
                
            # Plot the ball tracks on tactical map
            if len(ball_track_history['src'])>0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)
            
            # Combine annotated frame and tactical map in one image with colored border separation
            border_color = [255,255,255]                                                                        # Set border color (BGR)
            annotated_frame=cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,                                 # Add borders to annotated frame
                                                cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,                # Add borders to tactical map 
                                               value=border_color)

            # Ajouter des bordures blanches en haut et en bas de tac_map_copy
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # Redimensionner tac_map_copy pour avoir la même hauteur que annotated_frame
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))

            # Redimensionner dataframe_img pour avoir la même hauteur que annotated_frame
            dataframe_img = cv2.resize(dataframe_img, (dataframe_img.shape[1], annotated_frame.shape[0]))

            # Supprimer les deux premières et dernières colonnes de dataframe_img
            dataframe_img = dataframe_img[:, 40:-40, :]

            # Concaténer les images redimensionnées
            final_img = cv2.hconcat([dataframe_img, annotated_frame, tac_map_copy])            ## Add info annotation
            
            cv2.putText(final_img, "Video Insights", (190,190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(final_img, "Video after applying Yolo Model", (920,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(final_img, "Tactical Map", (1890,145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
    
            # Display possession percentage for each team
            cv2.putText(final_img, f"{data['team_home_name']} Possession: {team1:.0f}%", (550, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255 , 255), 2)
            cv2.putText(final_img, f"{data['team_away_name']} Possession: {team2:.0f}%", (550, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            new_frame_time = time.time()                                                                        # Get time after finished processing current frame
            fps = 1/(new_frame_time-prev_frame_time)                                                            # Calculate FPS as 1/(frame proceesing duration)
            prev_frame_time = new_frame_time                                                                    # Save current time to be used in next frame
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    
        ret, buffer = cv2.imencode('.jpg', final_img)
        final_img = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + final_img + b'\r\n')
    
    # Release the video capture object and close the display window
    cap.release()

global_data = {}

@app.route('/video_analysis')
def Video_Analysis():
    return render_template('video_analysis.html', out=your_count_name())

@app.route('/analyse_video', methods=['POST'])
def analyse_video():
    global global_data  # Utiliser la variable globale pour stocker les données

    # Récupération des données du formulaire
    team_home_name = request.form.get('text1')
    team_away_name = request.form.get('text2')
    team_home_color = request.form.get('color-picker')
    team_away_color = request.form.get('color-picker1')
    team_home_gk_color = request.form.get('color-picker3')
    team_away_gk_color = request.form.get('color-picker4')
    detection_confidence_threshold = request.form.get('range')
    palette_colors = request.form.get('range1')
    keypoints_confidence_threshold = request.form.get('range2')
    keypoints_rmse_tolerance = request.form.get('range3')
    ball_track_reset_threshold = request.form.get('adjustable-input')
    ball_track_distance_threshold = request.form.get('adjustable-input1')
    max_ball_track_length = request.form.get('adjustable-input2')
    show_color_palettes = 'show_color_palettes' in request.form
    show_player_detection = 'show_player_detection' in request.form
    show_ball_tracks = 'show_ball_tracks' in request.form
    show_keypoints_detections = 'show_keypoints_detections' in request.form

    # Initialisation du chemin du fichier vidéo
    video_path = None

    # Récupération du fichier vidéo
    video_file = request.files['input-file']
    if video_file and video_file.filename != '':
        video_filename = video_file.filename
        video_path = os.path.join('./static/', video_filename)
        video_file.save(video_path)

    # Stockage des données dans global_data
    global_data = {
        'team_home_name': team_home_name,
        'team_away_name': team_away_name,
        'team_home_color': hex_to_rgb(team_home_color),
        'team_away_color': hex_to_rgb(team_away_color),
        'team_home_gk_color': hex_to_rgb(team_home_gk_color),
        'team_away_gk_color': hex_to_rgb(team_away_gk_color),
        'detection_confidence_threshold': int(detection_confidence_threshold) / 100,
        'palette_colors': int(palette_colors),
        'keypoints_confidence_threshold': int(keypoints_confidence_threshold) / 100,
        'keypoints_rmse_tolerance': int(keypoints_rmse_tolerance),
        'ball_track_reset_threshold': int(ball_track_reset_threshold),
        'ball_track_distance_threshold': int(ball_track_distance_threshold),
        'max_ball_track_length': int(max_ball_track_length),
        'video_path': video_path,
        "show_color_palettes": show_color_palettes,
        "show_player_detection": show_player_detection,
        "show_ball_tracks": show_ball_tracks,
        "show_keypoints_detections": show_keypoints_detections,
    }

    return Response(generate_frames(global_data), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(global_data), mimetype='multipart/x-mixed-replace; boundary=frame')


#####################################################
# All other backend functionnalities like login ... #
#####################################################
@app.route('/index')
def index():
    return render_template('index.html', out=your_count_name())

@app.route("/")
def login():
    data = {
        'First Name': '',
        'Last Name': '',
        'Gender': '',
        'Age': '',
        'Email': '',
        'Password': '',
        'Status': '',
        'error_message': '',
    }
    return render_template('login.html', out =data)

@app.route('/register_open')
def register_open():
    data = {
        'First Name': '',
        'Last Name': '',
        'Gender': '',
        'Age': '',
        'Status': '',
        'error_message': ''
    }
    return render_template('register.html', out = data)

@app.route('/register', methods=['POST'])
def register():
    # Récupérer les données soumises par le formulaire
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    gender = request.form['gender']
    age = request.form['age']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    status = request.form['status']

    # Stocker les données dans un dictionnaire
    data = {
        'First Name': first_name,
        'Last Name': last_name,
        'Gender': gender,
        'Age': age,
        'Email': email,
        'Password': password,
        'Status': status
    }

    if check_email_exists(data['Email']): # Enregistrer les données dans un fichier CSV
        data['exist_account'] = f"This email ({data['Email']}) already exists. Please use a different email address."
        return render_template('register.html', out= data)
    elif password != confirm_password:
        data['error_message'] = "Password and confirmation do not match."
        return render_template('register.html', out= data)
    else:
        save_to_csv(data)
        return render_template('login.html', out= data)

@app.route('/test_login', methods=['POST'])
def test_login():
    email = request.form['email']
    password = request.form['password']

    with open('./static/login/data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Email'] == email and row['Password'] == password:
                # writing into the file
                user_name = {'First Name': [row['First Name']], 'Last Name': [row['Last Name']]}
                df = pd.DataFrame.from_dict(user_name)
                df.to_csv("./static/login/name.csv", index=False)

                return render_template('index.html', out=your_count_name())

    data = {
        "Email": email,
        "Password": password,
        "error_message": "Your email or Password is incorrect. Try Again!"
    }
    return render_template('login.html', out=data)


@app.route("/reset_password_page")
def reset_password_page():
    data ={
        'Email': '',
        'Password': ''
    }

    return render_template('reset_password.html', out = data)

@app.route("/reset_password", methods=["POST"])
def reset_password():
    email = request.form['email1']
    password = request.form['password1']
    confirm_password = request.form['confirm_password']
    
    # Stocker les données dans un dictionnaire
    data = {
        'Email': email,
        'Password': password,
    }
    
    if not check_email_exists(data['Email']): # Vérifier si l'e-mail existe dans la base de données
        data['exist_account'] = f"This email ({data['Email']}) does not exist in the database. Please enter a correct one."
        return render_template('reset_password.html', out= data)  # Renvoyer le formulaire de connexion avec le message d'erreur
    if password != confirm_password:
        data['error_message'] = "Password and confirmation do not match."
        return render_template('reset_password.html', out= data)  # Renvoyer le formulaire de connexion avec le message d'erreur
    else:
        update_csv(data)
        # Ici, vous pouvez implémenter la logique de réinitialisation de mot de passe
        return render_template('login.html', out= data)  # Renvoyer une page de confirmation de réinitialisation de mot de passe

# Convertir les valeurs int64 en types sérialisables en JSON
def convert_to_json_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(element) for element in obj]
    else:
        return obj

def delete_files_folder(repository):
    # Parcourir tous les fichiers du répertoire
    for file_name in os.listdir(repository):
        file_path = os.path.join(repository, file_name)
        # Vérifier si c'est un fichier
        if os.path.isfile(file_path):
            # Supprimer le fichier
            os.remove(file_path)

def check_email_exists(email):
    df = pd.read_csv('./static/login/data.csv')
    if email in df['Email'].values:
        return True
    return False

def save_to_csv(data):
    # Ajouter une nouvelle ligne au DataFrame
    new_data = pd.DataFrame(data, index=[0])
    new_data.to_csv('./static/login/data.csv', mode='a', header=not os.path.isfile('./static/login/data.csv'), index=False)

def update_csv(data):
    # reading the csv file 
    df = pd.read_csv("./static/login/data.csv", encoding='latin1') 
    
    #Find index of user
    id = df[df['Email'] == data['Email']].index[0]
    # updating the column value/data 
    df.loc[id, 'Password'] =data['Password']
  
    # writing into the file 
    df.to_csv("./static/login/data.csv", index=False) 

def your_count_name():
        if os.path.isfile("./static/login/name.csv"):
            df = pd.read_csv("./static/login/name.csv")
            name = df['First Name'][0] +" "+ df['Last Name'][0]
            return name
        return ""

if __name__ == '__main__':
    app.run(debug=True)  # Set debug mode to True