# Welcome to the UQAC Smart System for Sport Platform (UQACsss)

The UQACsss platform offers a comprehensive suite of Data Science solutions tailored for optimizing sports performance, particularly in football. It comprises a innovative solution aimed at addressing the requirements of teams, coaches, and sports analysts.

## Solution Name: Match Video Analysis for Extracting Insights (Statistics)

This solution utilizes Football Object Detection with Tactical Map Position Estimation to extract insights such as total distance covered, velocity, and match possession for each frame. It represents a full implementation of a real-time video processing and football analysis system, integrating machine learning techniques, object detection, and team prediction to provide in-depth analysis of on-field actions.

The system employs YOLOv8 object detection models to identify players, referees, and the ball in every video frame. Additionally, it utilizes landmark detection models to map the coordinates of detected objects onto a tactical map of the football field.

Furthermore, the code includes functionality to analyze possession during the match. By tracking the movement of the ball and players, it determines which team has possession of the ball at any given time and calculates the possession percentage for each team throughout the match. This possession analysis enhances the comprehensive football analysis provided by the system.

**Key Features**

1. **Object Detection:**
   - Utilization of the YOLOv8 model to detect players, referees, and the ball in each frame.
   - Utilization of the YOLOv8 model to detect key points of the field in each frame.

2. **Coordinate Transformation:**
   - Calculation of the homographic transformation matrix between the coordinates of the detected key points on the field and the corresponding coordinates on the tactical map.

3. **Player Team Prediction:**
   - Extraction of dominant color palettes for each detected player.
   - Comparison of color palettes with predefined team colors to assign a team to each player.

4. **Annotation of Images:**
   - Display of bounding boxes and labels for players, referees, and the ball.
   - Display of color palettes of players and predicted team names.
   - Display of player positions on the tactical map with corresponding team colors.
   - Display of the ball's trajectory history on the tactical map.

5. **Calculation of Total Distance Covered and Velocity:**
   - Tracking the movement of players to calculate the total distance covered by each player.
   - Derivation of player velocities based on their movements within successive frames.

6. **Ball Possession Analysis:**
   - Tracking the position of the ball and player's positions to determine possession.
   - Use of consecutive frames to determine continuous possession.
   - Comparison of possession times for both teams to determine current ball possession.

**Algorithms and Techniques:**

- **YOLOv8 Object Detection:** YOLOv8 (You Only Look Once version 8) is a real-time object detection algorithm that locates and classifies multiple objects in a single image. It is employed to detect players, referees, and the ball in football match video images.
- **Team Prediction based on Color and Position Analysis:** This technique involves analyzing the colors of player's jerseys and associating them with specific teams. By combining this analysis with player's positions on the field, the system predicts the teams.