#----------------------------------------------------------------------
# C O M P U T E R   V I S I O N   D U C K   T R A C K I N G
#----------------------------------------------------------------------


# Matricula: A01660619
# Nombre:    JEFFRY JOHNSON


#----------------------------------------------------------------------
# L I B R E R I A S
#----------------------------------------------------------------------

import cv2
import glob
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
from ultralytics import YOLO


#----------------------------------------------------------------------
# C L A S E   K A L M A N   F I L T E R
#----------------------------------------------------------------------

class KalmanFilter:

    #----------------------------------------------------------------------
    # FUNCTION INIT_KALMAN_FILTER
    #----------------------------------------------------------------------

    def __init__(self, delta_t: float = 1.0):
        """
        Construct a 2D Kalman Filter with state [x, y, vx, vy].

        Args:
            delta_t: Time interval between frames
        """
        # 1) SET TIME STEP
        self.dt = delta_t

        # 2) INITIAL STATE VECTOR [x, y, vx, vy]
        self.state = np.zeros(4)

        # 3) STATE TRANSITION MATRIX (A)
        self.A = np.array([
            [1, 0, delta_t, 0],
            [0, 1, 0, delta_t],
            [0, 0, 1,       0],
            [0, 0, 0,       1]
        ])

        # 4) MEASUREMENT MATRIX (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 5) PROCESS NOISE COVARIANCE (Q)
        self.Q = np.eye(4) * 0.1

        # 6) MEASUREMENT NOISE COVARIANCE (R)
        self.R = np.eye(2) * 1.0

        # 7) ESTIMATE ERROR COVARIANCE (P)
        self.P = np.eye(4) * 1000


    #----------------------------------------------------------------------
    # FUNCTION FORECAST_STATE
    #----------------------------------------------------------------------

    def forecast_state(self) -> np.ndarray:
        """
        Forecast the next state using the motion model.

        Returns:
            np.ndarray: Predicted position vector [x, y].
        """
        # 1) PROPAGATE STATE: x_k = A · x_{k-1}
        self.state = self.A @ self.state

        # 2) PROPAGATE COVARIANCE: P_k = A · P_{k-1} · Aᵀ + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        # 3) RETURN MEASURED POSITION: z = H · x_k
        return self.H @ self.state


    #----------------------------------------------------------------------
    # FUNCTION CORRECT_STATE
    #----------------------------------------------------------------------

    def correct_state(self, obs: np.ndarray) -> None:
        """
        Correct the state estimate using a new observation.

        Args:
            obs (np.ndarray): Observation vector [x, y].
        """
        # 1) Compute innovation covariance: S = H·P·Hᵀ + R
        S = self.H @ self.P @ self.H.T + self.R
        # 2) Compute Kalman gain: K = P·Hᵀ·S⁻¹
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 3) Compute innovation (measurement residual): y = z - H·x
        innovation = obs - (self.H @ self.state)

        # 4) Update state estimate: x = x + K·y
        self.state = self.state + K @ innovation

        # 5) Update error covariance: P = (I - K·H)·P
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P



#----------------------------------------------------------------------
# C L A S E   D U C K   T R A C K E R   
#----------------------------------------------------------------------

class Duck_Tracker:

    #----------------------------------------------------------------------
    # FUNCTION INIT
    #----------------------------------------------------------------------

    def __init__(self,
             model_adress: str = '/home/strpicket/Duck_vision/yolo.pt',  # Ruta al archivo de pesos del modelo YOLOv8
             min_detection: float = 0.3,                                     # Umbral mínimo de confianza para aceptar una detección
             min_lenght_pixels: int = 20,                                    # Distancia mínima en píxeles entre detecciones para considerarlas distintas
             max_fps: int = 2083,                                            # Número máximo de fotogramas a procesar antes de detenerse
             max_ducks: int = 7,                                             # Número máximo de patos que se pueden rastrear simultáneamente
             last_positions: int = 30,                                       # Número de posiciones pasadas que se guardan para cada pista
             max_pixels_lane: int = 50,                                      # Distancia máxima en píxeles para asociar una detección a una pista existente
             fps_tracking: int = 20,                                         # Fotogramas sin detección antes de marcar una pista como perdida
             lane_threshold: float = 0.3,                                    # Umbral de puntuación para reidentificar una pista perdida
             mat_summary: bool = True,                                       # Indica si generar un resumen final con Matplotlib
             keyframes: List[int] = None                                     # Lista de índices de fotogramas donde guardar keyframes
            ):
        """
        Initialize the advanced duck tracker with trajectory-based ID tracking,
        plus options for Matplotlib summary & keyframes.
        """
        if not os.path.exists(model_adress):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_adress}")
            
        self.model = YOLO(model_adress)                                        # YOLO model for duck detection

        # Original attributes:
        self.duck_positions = {}                                               # Current positions of each duck in the frame
        self.duck_history = {}                                                 # Historical positions across all processed frames
        self.min_detection = min_detection                                     # Minimum confidence to accept a detection
        self.min_lenght_pixels = min_lenght_pixels                             # Minimum pixel distance to treat detections as separate
        self.frame_count = 0                                                   # Counter of processed frames
        self.max_fps = max_fps                                                 # Maximum number of frames to process
        self.max_ducks = max_ducks                                             # Maximum number of ducks trackable at once
        self.grid_size = (8, 8)                                                # Grid dimensions for spatial partitioning
        self.all_frames_data = {}                                              # Per-frame tracking data for export
        self.last_positions = last_positions                                   # Number of past positions to keep per track
        self.max_pixels_lane = max_pixels_lane                                 # Max pixel distance to associate detection to existing track
        self.fps_tracking = fps_tracking                                       # Frames without detection before marking track as lost
        self.lane_threshold = lane_threshold                                   # Threshold score for re-identifying a lost track
        self.available_ids = [f"duck_{i+1}" for i in range(self.max_ducks)]    # Pool of pre-assigned duck IDs
        self.id_in_use = {i: False for i in self.available_ids}                # Which IDs are currently active
        self.tracked_ducks = {}                                                # Active tracks with Kalman filters
        self.next_duck_id = 1                                                  # Fallback counter for assigning new IDs
        self.lost_ducks = {}                                                   # Tracks marked as lost awaiting re-ID
        self.id_persistency = 10                                               # Frames to retain an ID after last detection
        self.color_stability = {}                                              # Historical color info to avoid rapid color changes
        self.color_map = {                                                     # Mapping duck color to plot/text/RGB styling
            'yellow': {'plot': 'y', 'text': 'black', 'rgb': (1, 1, 0)},
            'black':  {'plot': 'k', 'text': 'white', 'rgb': (0, 0, 0)},
        }
        # New options:
        self.mat_summary = mat_summary                                         # Whether to generate final Matplotlib summary
        self.keyframes = set(keyframes) if keyframes else set()                # Frame indices where keyframes are saved
        self.last_frame = None                                                 # Placeholder for last processed frame

    #----------------------------------------------------------------------
    # FUNCTION DUCK TRACKING INIT
    #----------------------------------------------------------------------

    def init_duck_tracking(self, position: Tuple[int, int], color: str) -> str:
        """
        Initialize tracking for a new duck with Kalman filter.
        Uses pre-assigned IDs and avoids creating more than the maximum.
        """
        # Try to find an unused ID
        for duck_id in self.available_ids:                             # self.available_ids: list of pre-assigned duck IDs
            if not self.id_in_use[duck_id]:                            # self.id_in_use: tracks which IDs are active
                self.id_in_use[duck_id] = True                         # mark this ID as now in use
                
                # Initialize a new Kalman filter for this duck
                kf = KalmanFilter()
                kf.state[:2] = position                                # set initial state based on position
                
                # Create a new tracking entry
                self.tracked_ducks[duck_id] = {                        # self.tracked_ducks: dict of active track data
                    'positions': deque(maxlen=self.last_positions),    # self.last_positions: history length per track
                    'color': color,                                    # store detected color
                    'last_seen': self.frame_count,                     # self.frame_count: current frame index
                    'trajectory': [],                                  # list to record full trajectory
                    'kalman_filter': kf,                               # attach Kalman filter instance
                    'tracking_score': 1.0,                             # initial confidence score
                    'persistence_count': self.id_persistency           # self.id_persistency: frames to keep ID after loss
                }
                self.tracked_ducks[duck_id]['positions'].append(position)
                self.tracked_ducks[duck_id]['trajectory'].append(position)
                
                return duck_id

        # If all IDs are in use, pick the one with the lowest tracking score
        min_score = float('inf')
        replaced_id = None
        for duck_id, data in self.tracked_ducks.items():
            if data['tracking_score'] < min_score:
                min_score = data['tracking_score']
                replaced_id = duck_id

        if replaced_id:
            # Reuse this ID for the new duck
            kf = KalmanFilter()
            kf.state[:2] = position
            
            self.tracked_ducks[replaced_id] = {
                'positions': deque(maxlen=self.last_positions),
                'color': color,
                'last_seen': self.frame_count,
                'trajectory': [],
                'kalman_filter': kf,
                'tracking_score': 0.7,                               # slightly lower for replaced track
                'persistence_count': self.id_persistency
            }
            self.tracked_ducks[replaced_id]['positions'].append(position)
            self.tracked_ducks[replaced_id]['trajectory'].append(position)
            
            return replaced_id

        # Fallback (should not happen)
        return "duck_1"


    #----------------------------------------------------------------------
    # FUNCTION UPDATE DUCK TRACKING
    #----------------------------------------------------------------------

    def update_duck_tracking(self, duck_id: str, position: Tuple[int, int], color: str, confidence: float) -> str:
        """
        Update tracking information for an existing duck.
        """
        # If this duck ID is not already tracked, initialize it instead
        if duck_id not in self.tracked_ducks:                                     
            return self.init_duck_tracking(position, color)                      

        # Retrieve the track data dictionary for this duck
        track_data = self.tracked_ducks[duck_id]                                 

        # ---------- KALMAN FILTER UPDATE ----------
        kf = track_data['kalman_filter']                                         # Kalman filter instance for this duck
        measurement = np.array(position)                                         # Convert tuple to NumPy array
        kf.correct_state(measurement)                                            # Perform measurement update

        # ---------- POSITION & TRAJECTORY ----------
        track_data['positions'].append(position)                                 # Append to recent positions deque
        track_data['trajectory'].append(position)                                # Append to full trajectory list
        track_data['last_seen'] = self.frame_count                               # Record frame index when last seen

        # ---------- COLOR STABILITY ----------
        # Only update color if it matches previous or confidence is high
        if track_data['color'] == color or confidence > 0.7:                     
            track_data['color'] = color                                          # Update tracked color

        # ---------- PERSISTENCE RESET ----------
        track_data['persistence_count'] = self.id_persistency                    # Reset lost-track persistence counter

        # ---------- TRACKING SCORE SMOOTHING ----------
        # Blend previous score with current confidence for stability
        prev_score = track_data['tracking_score']                                # Previous tracking confidence
        track_data['tracking_score'] = min(
            1.0,
            prev_score * 0.8 + confidence * 0.2
        )

        return duck_id                                                           # Return the duck's ID


    #----------------------------------------------------------------------
    # FUNCTION CALCULATE TRAJECTORY SIMILARITY
    #----------------------------------------------------------------------

    def calculate_trajectory_similarity(self, track1: List[Tuple[int, int]], track2: List[Tuple[int, int]]) -> float:
        """
        Calculate similarity between two trajectories using Dynamic Time Warping (DTW).
        """
        # If either trajectory is empty, return zero similarity
        if not track1 or not track2:                                            
            return 0.0                                                         

        # Convert trajectories to NumPy arrays for vectorized distance computation
        t1 = np.array(track1)  # shape: (n, 2)
        t2 = np.array(track2)  # shape: (m, 2)

        # Compute pairwise Euclidean distance matrix
        distances = cdist(t1, t2)  # shape: (n, m)

        # Prepare DTW cost matrix with infinities
        n, m = distances.shape                                               
        dtw_matrix = np.full((n + 1, m + 1), np.inf)                          
        dtw_matrix[0, 0] = 0.0  # starting point

        # Populate DTW matrix using recurrence relation
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = distances[i - 1, j - 1]                                
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],    # insertion
                    dtw_matrix[i, j - 1],    # deletion
                    dtw_matrix[i - 1, j - 1] # match
                )

        # Normalize by the length of the longer trajectory and the pixel threshold
        max_length = max(n, m)                                               
        # self.max_pixels_lane: maximum pixel distance threshold for associating detections
        normalized_cost = dtw_matrix[n, m] / (max_length * self.max_pixels_lane)  
        similarity = 1.0 - normalized_cost                                   

        # Clamp result to [0.0, 1.0]
        return max(0.0, min(1.0, similarity))                                


    #----------------------------------------------------------------------
    # FUNCTION FIND LOST DUCK MATCH
    #----------------------------------------------------------------------

    def find_lost_duck_match(self, detection: Dict, current_pos: Tuple[int, int]) -> Optional[str]:
        """
        Try to match a current detection to a previously lost duck based on trajectory,
        position prediction, color similarity, and time penalty.

        Args:
            detection: Dict with keys 'bbox', 'confidence', 'color' for current detection
            current_pos: Tuple[int, int] representing current (x, y) position of the detection

        Returns:
            Optional[str]: ID of the matched lost duck, or None if no match is found
        """
        best_match = None
        best_score = self.lane_threshold  # threshold for re-identifying a lost track

        # Iterate over all lost ducks to compute a matching score
        for lost_id, lost_data in self.lost_ducks.items():
            # Compute color match factor (1.0 if same color, otherwise 0.5)
            color_match = 1.0 if lost_data['color'] == detection['color'] else 0.5

            # Compute trajectory similarity using DTW against just the current position
            trajectory_similarity = self.calculate_trajectory_similarity(
                lost_data['trajectory'],
                [current_pos]
            )

            # Predict the next position with Kalman filter
            kf = lost_data['kalman_filter']
            predicted_pos = kf.forecast_state()

            # Compute normalized position similarity
            distance = np.linalg.norm(np.array(current_pos) - predicted_pos)
            position_similarity = 1.0 - min(1.0, distance / self.max_pixels_lane)  
            # self.max_pixels_lane: maximum pixel distance to associate detections

            # Compute time penalty based on how many frames since last seen
            frames_lost = self.frame_count - lost_data['last_seen']
            time_penalty = max(0, 1 - (frames_lost / (self.fps_tracking * 2)))  
            # self.fps_tracking: frames without detection before marking lost

            # Combine all factors into a single score
            combined_score = (
                0.3 * trajectory_similarity +
                0.4 * position_similarity +
                0.2 * time_penalty +
                0.1 * color_match
            )

            # Update best match if this score exceeds our threshold
            if combined_score > best_score:
                best_score = combined_score
                best_match = lost_id

        return best_match


    #----------------------------------------------------------------------
    # FUNCTION UPDATE_LOST_DUCKS
    #----------------------------------------------------------------------

    def update_lost_ducks(self) -> None:
        """
        Actualiza la lista de patos perdidos y mantiene IDs consistentes.
        Ahora guarda también la deque de posiciones para poder restaurarla.
        """
        # 1) Iterar sobre los tracks activos y decrementar su persistencia
        for duck_id, track_data in list(self.tracked_ducks.items()):
            # self.frame_count: contador de frames procesados hasta ahora
            # track_data['last_seen']: frame en que el pato fue visto por última vez
            frames_since_last = self.frame_count - track_data['last_seen']
            
            if frames_since_last > 0:
                # self.id_persistency: nº de frames que mantenemos un ID tras perder la detección
                # track_data['persistence_count']: contador de persistencia restante
                track_data['persistence_count'] = track_data.get('persistence_count', self.id_persistency) - 1  
                # track_data['tracking_score']: confiabilidad actual del track, se atenúa gradualmente
                track_data['tracking_score'] *= 0.95  
            
            # Si supera el timeout de tracking y la persistencia llega a 0, lo pasamos a lost_ducks
            # self.fps_tracking: nº de frames sin detección antes de considerar perdido un pato
            if frames_since_last > self.fps_tracking and track_data.get('persistence_count', 0) <= 0:
                self.lost_ducks[duck_id] = {
                    # self.last_positions: tamaño máximo del historial de posiciones
                    'positions':     deque(track_data['positions'], maxlen=self.last_positions),  
                    'trajectory':    list(track_data['trajectory']),         # copia de la trayectoria completa
                    'color':         track_data['color'],                   # color actual del pato
                    'kalman_filter': track_data['kalman_filter'],           # filtro de Kalman asociado
                    'last_seen':     track_data['last_seen'],               # último frame visto
                    'tracking_score':track_data['tracking_score'],          # score al momento de perder el track
                    'persistence_count': track_data.get('persistence_count', self.id_persistency)
                }
                # self.id_in_use: indica qué IDs de pato están actualmente asignados
                self.id_in_use[duck_id] = False  
                # eliminar de los tracks activos
                del self.tracked_ducks[duck_id]

        # 2) Limpiar lost_ducks muy antiguos para no acumular memoria
        for duck_id, lost_data in list(self.lost_ducks.items()):
            # Si ha pasado más de fps_tracking*4 frames desde que se perdió, lo eliminamos
            if self.frame_count - lost_data['last_seen'] > self.fps_tracking * 4:
                del self.lost_ducks[duck_id]


    #----------------------------------------------------------------------
    # FUNCTION MATCH_DUCKS_TO_DETECTIONS
    #----------------------------------------------------------------------

    def match_ducks_to_detections(self, detections: List[Dict]) -> Dict[str, Dict]:
        """
        Matches current detections to existing tracked ducks using the Hungarian algorithm.
        Maintains consistent IDs and reintegrates lost track positions.
        """
        # 1) If there are no detections, return an empty mapping
        if not detections:
            return {}

        # 2) Limit the number of detections to track to self.max_ducks  # maximum simultaneous ducks
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:self.max_ducks]

        # 3) Update the lost ducks list before matching  # cleans up and moves expired tracks
        self.update_lost_ducks()

        # 4) Build lists of active tracks and predict their next positions
        active_tracks = []
        predicted_positions = []
        for duck_id, track_data in self.tracked_ducks.items():
            frames_since = self.frame_count - track_data['last_seen']  # frames since last seen
            # consider track active if within fps_tracking or still has persistence_count
            if frames_since <= self.fps_tracking or track_data.get('persistence_count', 0) > 0:
                active_tracks.append(duck_id)
                predicted_positions.append(track_data['kalman_filter'].forecast_state())  # predict next position

        matched_ducks = {}
        used_detections = set()

        # 5) Attempt to re-identify lost ducks
        for i, det in enumerate(detections):
            center = self.calculate_center_point(det['bbox'])  # compute bounding‐box center
            lost_id = self.find_lost_duck_match(det, center)   # try to match a lost duck
            if lost_id:
                lost_data = self.lost_ducks.pop(lost_id)
                # restore full track for this lost duck
                self.tracked_ducks[lost_id] = {
                    'positions':         lost_data['positions'],         # historical positions deque
                    'trajectory':        lost_data['trajectory'],        # full trajectory list
                    'color':             lost_data['color'],             # last known color
                    'kalman_filter':     lost_data['kalman_filter'],     # restored Kalman filter
                    'last_seen':         self.frame_count,              # mark as seen this frame
                    'tracking_score':    lost_data['tracking_score'],    # restored tracking score
                    'persistence_count': self.id_persistency            # reset persistence counter
                }
                self.id_in_use[lost_id] = True  # mark this ID as in use again
                matched_ducks[lost_id] = det
                used_detections.add(i)

        # 6) Use the Hungarian algorithm to assign remaining detections
        if active_tracks and len(matched_ducks) < min(len(detections), self.max_ducks):
            remaining = [d for j, d in enumerate(detections) if j not in used_detections]
            rem_centers = [self.calculate_center_point(d['bbox']) for d in remaining]
            if rem_centers:
                cost = np.zeros((len(active_tracks), len(rem_centers)))
                for a, duck_id in enumerate(active_tracks):
                    for b, rc in enumerate(rem_centers):
                        pred = predicted_positions[a]              # predicted position from Kalman filter
                        dist = np.linalg.norm(pred - rc)          # Euclidean distance
                        # add a heavy penalty if colors differ
                        color_pen = 10 if self.tracked_ducks[duck_id]['color'] != remaining[b]['color'] else 0
                        cost[a, b] = dist + color_pen
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    # only accept assignments within a reasonable distance
                    if cost[r, c] <= self.max_pixels_lane * 1.5:  # max_pixels_lane: association threshold
                        duck_id = active_tracks[r]
                        det = remaining[c]
                        center = self.calculate_center_point(det['bbox'])
                        self.update_duck_tracking(duck_id, center, det['color'], det['confidence'])
                        matched_ducks[duck_id] = det
                        used_detections.add(c)

        # 7) Initialize new tracks for any leftover detections
        slots = self.max_ducks - len(self.tracked_ducks)  # available tracking slots
        for j, det in enumerate(detections):
            if j in used_detections or slots <= 0:
                continue
            center = self.calculate_center_point(det['bbox'])
            new_id = self.init_duck_tracking(center, det['color'])  # start a new track
            matched_ducks[new_id] = det
            slots -= 1

        return matched_ducks


    #----------------------------------------------------------------------
    # FUNCTION PROCESS_FRAME
    #----------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single video frame: detect ducks, match to existing tracks,
        update each track, and record positions.
        """
        # 1) Detect ducks in the current frame
        detections = self.detect_ducks(frame)

        # 2) Clear previous frame positions before updating
        self.duck_positions.clear()       # dict[str, dict]: holds positions for this frame

        # 3) Match detections to existing tracked ducks
        matched_ducks = self.match_ducks_to_detections(detections)

        # 4) For each matched duck, update tracking and record data
        for duck_id, detection in matched_ducks.items():
            # a) Compute the center point of the bounding box
            center = self.calculate_center_point(detection['bbox'])
            # b) Compute grid cell coordinates for this center
            grid_pos = self.calculate_grid_position(center, frame.shape)

            # c) Update the duck's Kalman filter and track info
            self.update_duck_tracking(
                duck_id,
                center,
                detection['color'],
                detection['confidence']
            )

            # d) Compute the current velocity from recent positions
            velocity = self.calculate_velocity(duck_id)

            # e) Store all relevant info for this duck in this frame
            self.duck_positions[duck_id] = {
                'position':      center,                              # (x, y) pixel coords
                'grid_position': grid_pos,                            # (grid_x, grid_y) cell coords
                'confidence':    detection['confidence'],              # float detection confidence
                'bbox':          detection['bbox'],                    # (x1, y1, x2, y2)
                'color':         detection['color'],                   # 'yellow' or 'black'
                'velocity':      velocity,                            # (vx, vy) or None
                'trajectory':    list(self.tracked_ducks[duck_id]['positions'])  # recent positions
            }

        # 5) Increment the global frame counter
        self.frame_count += 1           # int: total frames processed so far


    #----------------------------------------------------------------------
    # FUNCTION CALCULATE_VELOCITY
    #----------------------------------------------------------------------

    def calculate_velocity(self, duck_id: str) -> Optional[Tuple[float, float]]:
        """
        Calculate the instantaneous velocity of a tracked duck using its two most recent positions.
        """
        # 1) Verify that this duck is being tracked and has at least two recorded positions
        if duck_id not in self.tracked_ducks or len(self.tracked_ducks[duck_id]['positions']) < 2:
            return None  # Not enough data to compute velocity

        # 2) Access the deque of recent positions for this duck
        positions = self.tracked_ducks[duck_id]['positions']   # deque[(x, y), ...]

        # 3) Guard again against insufficient positions (defensive)
        if len(positions) < 2:
            return None

        # 4) Extract the last two positions
        prev_pos = positions[-2]   # second-to-last recorded (x, y)
        last_pos = positions[-1]   # most recent recorded (x, y)

        # 5) Compute frame-to-frame displacement as velocity vector
        vx = last_pos[0] - prev_pos[0]   # horizontal velocity (pixels/frame)
        vy = last_pos[1] - prev_pos[1]   # vertical velocity (pixels/frame)

        # 6) Return the velocity tuple
        return (vx, vy)


    #----------------------------------------------------------------------
    # FUNCTION VISUALIZE_2D
    #----------------------------------------------------------------------

    def visualize_2d(self, frame: np.ndarray) -> None:
        """
        Visualize tracked ducks and their Kalman filter predictions in a 2D layout.
        """
        # 1) Create a new figure with two subplots side by side
        plt.figure(figsize=(15, 7))  

        # ----- Subplot 1: Original video frame with overlays -----
        ax1 = plt.subplot(121)  
        # 1.1) Show the BGR frame as RGB
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
        # 1.2) Draw the grid on top of the image
        self.draw_grid(ax1, frame.shape)  

        # 1.3) Plot lost ducks first, in gray with dashed trajectories
        for lost_id, lost_data in self.lost_ducks.items():  
            if lost_data['trajectory']:
                last_pos = lost_data['trajectory'][-1]  
                # 1.3.1) Plot the past trajectory in gray
                traj = np.array(lost_data['trajectory'])  
                plt.plot(traj[:, 0], traj[:, 1], '--', color='gray', alpha=0.3, linewidth=1)  
                # 1.3.2) Predict next position via Kalman filter and plot as red plus
                pred = lost_data['kalman_filter'].forecast_state()  
                plt.plot(pred[0], pred[1], 'r+', markersize=8, alpha=0.3)  
                # 1.3.3) Label the lost duck at its last known position
                plt.text(last_pos[0], last_pos[1] - 10, f'{lost_id} (lost)', 
                        color='gray', fontsize=8, bbox=dict(facecolor='gray', alpha=0.3))

        # 1.4) Plot active ducks with bounding boxes, trajectories, and labels
        for duck_id, data in self.duck_positions.items():  
            # 1.4.1) Unpack bounding box and position
            x1, y1, x2, y2 = data['bbox']  
            x, y = data['position']  
            confidence = data['confidence']  
            duck_color = data['color']  
            velocity = data['velocity']  
            grid_pos = data['grid_position']  
            score = self.tracked_ducks[duck_id]['tracking_score']  

            # 1.4.2) Retrieve RGB and text colors from the map
            cmap = self.color_map[duck_color]  

            # 1.4.3) Draw the bounding box with transparency based on tracking score
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                color=cmap['rgb'], alpha=score, linewidth=2)  
            plt.gca().add_patch(rect)

            # 1.4.4) Draw the past trajectory with fading alpha
            traj = np.array(data['trajectory'])  
            for i in range(len(traj) - 1):
                alpha = (i + 1) / len(traj) * 0.8  
                plt.plot(traj[i:i+2, 0], traj[i:i+2, 1], '--', 
                        color=cmap['rgb'], alpha=alpha, linewidth=1)

            # 1.4.5) Plot center point of the duck
            plt.plot(x, y, cmap['plot'] + 'o', markersize=8)  

            # 1.4.6) Predict next position and plot
            pred = self.tracked_ducks[duck_id]['kalman_filter'].forecast_state()  
            plt.plot(pred[0], pred[1], 'r+', markersize=8, alpha=0.6)  

            # 1.4.7) Build label text with ID, confidence, color, grid, score, and velocity
            label = (f"{duck_id} ({confidence:.2f})\n"
                    f"Color: {duck_color}\n"
                    f"Grid: {grid_pos}\n"
                    f"Score: {score:.2f}")
            if velocity:
                label += f"\nV: ({velocity[0]:.1f}, {velocity[1]:.1f})"

            # 1.4.8) Place the label above the bounding box
            plt.text(x1, y1 - 10, label, color=cmap['text'], fontsize=8,
                    bbox=dict(facecolor=cmap['rgb'], alpha=0.7))

        plt.title('Detections and Kalman Predictions')  
        plt.axis('off')  

        # ----- Subplot 2: Blank canvas with trajectories only -----
        ax2 = plt.subplot(122)  
        plt.imshow(np.zeros_like(frame[:, :, 0]), cmap='gray')  
        self.draw_grid(ax2, frame.shape)  

        # 2.1) Plot lost duck trajectories in gray
        for lost_id, lost_data in self.lost_ducks.items():
            if lost_data['trajectory']:
                traj = np.array(lost_data['trajectory'])
                plt.plot(traj[:, 0], traj[:, 1], '--', color='gray', alpha=0.3, linewidth=1)

        # 2.2) Plot active duck trajectories with fade and predicted arrows
        for duck_id, data in self.duck_positions.items():
            traj = np.array(data['trajectory'])
            cmap = self.color_map[data['color']]
            score = self.tracked_ducks[duck_id]['tracking_score']

            # 2.2.1) Trajectory line with fade
            for i in range(len(traj) - 1):
                alpha = (i + 1) / len(traj) * score
                plt.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=cmap['rgb'], alpha=alpha)

            # 2.2.2) Current and predicted positions
            x, y = data['position']
            plt.scatter(x, y, c=[cmap['rgb']], s=100, alpha=score)
            pred = self.tracked_ducks[duck_id]['kalman_filter'].forecast_state()
            plt.plot([x, pred[0]], [y, pred[1]], 'r--', alpha=0.4)
            plt.scatter(pred[0], pred[1], c='r', s=50, alpha=0.4)

            # 2.2.3) Annotate with minimal info
            info = f"{duck_id}\n({data['color']})\nScore: {score:.2f}"
            plt.annotate(info, (x, y), xytext=(5, 5), textcoords='offset points',
                        color='white', fontsize=8)

        plt.title('2D Trajectories and Predictions')  
        plt.axis('off')  
        plt.tight_layout()  
        plt.show()  


    #----------------------------------------------------------------------
    # FUNCTION GET_DUCK_COLOR
    #----------------------------------------------------------------------

    def get_duck_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Determine the duck's color by analyzing its region of interest.

        Args:
            frame (np.ndarray): The current video frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).

        Returns:
            str: 'yellow' if the duck is predominantly yellow, otherwise 'black'.
        """
        # 1) Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox  

        # 2) Extract the region of interest (ROI) from the frame
        roi = frame[y1:y2, x1:x2]  

        # 3) Convert the ROI from BGR to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  

        # 4) Define HSV thresholds for yellow and create a mask
        lower_yellow = np.array([20, 100, 100])  
        upper_yellow = np.array([30, 255, 255])  
        yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)  

        # 5) Calculate the ratio of yellow pixels within the ROI
        yellow_ratio = np.sum(yellow_mask > 0) / (yellow_mask.size)  

        # 6) Return 'yellow' if ratio exceeds threshold, else 'black'
        return 'yellow' if yellow_ratio > 0.15 else 'black'


    #----------------------------------------------------------------------
    # FUNCTION CALCULATE_GRID_POSITION
    #----------------------------------------------------------------------

    def calculate_grid_position(self, point: Tuple[int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate the grid cell coordinates corresponding to an image point.

        Args:
            point (Tuple[int, int]): (x, y) pixel coordinates in the image.
            frame_shape (Tuple[int, int]): Frame dimensions as (height, width).

        Returns:
            Tuple[int, int]: (grid_x, grid_y) cell indices in the defined grid.
        """
        # 1) Unpack the image point coordinates
        x, y = point  # x: horizontal pixel, y: vertical pixel

        # 2) Retrieve frame dimensions
        height, width = frame_shape[:2]  # height: number of rows, width: number of columns

        # 3) Compute horizontal grid index
        # self.grid_size[0] is the number of columns in the grid
        grid_x = int((x / width) * self.grid_size[0])

        # 4) Compute vertical grid index
        # self.grid_size[1] is the number of rows in the grid
        grid_y = int((y / height) * self.grid_size[1])

        # 5) Return the grid cell coordinates
        return (grid_x, grid_y)


    #----------------------------------------------------------------------
    # FUNCTION CALCULATE_CENTER_POINT
    #----------------------------------------------------------------------

    def calculate_center_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate the center point of a bounding box.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box as (x1, y1, x2, y2).

        Returns:
            Tuple[int, int]: Center coordinates (x_center, y_center).
        """
        # 1) Unpack bounding box coordinates
        x1, y1, x2, y2 = bbox  # x1,y1: top-left; x2,y2: bottom-right corners

        # 2) Compute center X coordinate
        # Average of left and right X values
        x_center = int((x1 + x2) / 2)

        # 3) Compute center Y coordinate
        # Average of top and bottom Y values
        y_center = int((y1 + y2) / 2)

        # 4) Return the center point
        return (x_center, y_center)

    #----------------------------------------------------------------------
    # FUNCTION DETECT_DUCKS
    #----------------------------------------------------------------------

    def detect_ducks(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect ducks in a video frame using YOLOv8.

        Args:
            frame (np.ndarray): Input video frame in BGR color space.

        Returns:
            List[Dict]: A list of detections, each containing:
                - 'bbox': Tuple[int, int, int, int] bounding box (x1, y1, x2, y2)
                - 'confidence': float confidence score
                - 'color': str estimated duck color ('yellow' or 'black')
        """
        # 1) Run YOLOv8 inference on the frame
        results = self.model(frame)  # self.model: pretrained YOLOv8 for duck detection

        detections: List[Dict] = []

        for result in results:
            boxes = result.boxes  # list of prediction boxes

            # 2) Build a list of raw detections above confidence threshold
            raw_detections: List[Dict] = []
            for box in boxes:
                # Only consider class 0 (duck) and confidence > min_detection
                if box.cls == 0 and float(box.conf[0]) > self.min_detection:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    raw_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })

            # 3) Keep only the top-N most confident detections
            raw_detections = sorted(
                raw_detections,
                key=lambda d: d['confidence'],
                reverse=True
            )[:self.max_ducks]  # self.max_ducks: maximum ducks to track

            # 4) Filter out detections that are too close, unless color differs
            filtered: List[Dict] = []
            centers: List[Tuple[int, int]] = []

            for det in raw_detections:
                center = self.calculate_center_point(det['bbox'])  # compute bbox center

                if not centers:
                    # First detection always accepted
                    det['color'] = self.get_duck_color(frame, det['bbox'])  # estimate color
                    filtered.append(det)
                    centers.append(center)
                    continue

                # Compute distances to previously accepted centers
                distances = cdist([center], centers)[0]  # Euclidean distances

                if np.min(distances) > self.min_lenght_pixels:
                    # Far enough from others
                    det['color'] = self.get_duck_color(frame, det['bbox'])
                    filtered.append(det)
                    centers.append(center)
                else:
                    # If too close, but color differs, accept if slots remain
                    nearest_idx = np.argmin(distances)
                    current_color = self.get_duck_color(frame, det['bbox'])
                    if current_color != filtered[nearest_idx]['color']:
                        det['color'] = current_color
                        if len(filtered) < self.max_ducks:
                            filtered.append(det)
                            centers.append(center)

            detections = filtered

        return detections


    #----------------------------------------------------------------------
    # FUNCTION DRAW_GRID
    #----------------------------------------------------------------------

    def draw_grid(self, ax, shape: Tuple[int, int]):
        """
        Draw a grid overlay on the given Matplotlib axis.

        Args:
            ax: Matplotlib Axes object to draw the grid on.
            shape: Tuple[int, int] containing (height, width) of the frame.
        """
        height, width = shape[:2]  # unpack frame dimensions

        # Draw vertical grid lines
        for col in range(self.grid_size[0] + 1):  # self.grid_size[0]: number of grid columns
            x = (width / self.grid_size[0]) * col  # x-coordinate for this vertical line
            ax.plot([x, x], [0, height], 'w-', alpha=0.3)  # white line, 30% opacity

        # Draw horizontal grid lines
        for row in range(self.grid_size[1] + 1):  # self.grid_size[1]: number of grid rows
            y = (height / self.grid_size[1]) * row  # y-coordinate for this horizontal line
            ax.plot([0, width], [y, y], 'w-', alpha=0.3)  # white line, 30% opacity


    #----------------------------------------------------------------------
    # FUNCTION SAVE_POSITIONS
    #----------------------------------------------------------------------

    def save_positions(self, filename: str = None) -> None:
        """
        Save current duck positions and tracking history to a JSON file.

        Args:
            filename (str, optional): Output filename for saved positions. 
                                    If None, a timestamped name will be generated.
        """
        # Generate timestamped filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # current timestamp
            filename = f"duck_positions_{timestamp}.json"

        # Prepare data to save
        data_to_save = {
            'frame_count':    self.frame_count,    # number of frames processed so far
            'grid_size':      self.grid_size,      # dimensions of the spatial grid
            'ducks':          self.duck_positions, # latest positions and metadata for each duck
            'history':        self.duck_history    # stored history of all past positions
        }

        # Write JSON to disk
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)


    #----------------------------------------------------------------------
    # FUNCTION PROCESS_VIDEO
    #----------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str = "output",
        save_frames: bool = True,
        display: bool = False,
        save_originals: bool = True
    ) -> None:
        """
        Process an entire video and save the tracking results:
        1) Overlay lightweight tracking using OpenCV
        2) Save keyframes with Matplotlib when requested
        3) Generate a final summary plot with Matplotlib
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Open video file for reading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Read video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # total number of frames in input
        fps_input    = int(cap.get(cv2.CAP_PROP_FPS))            # frames per second of input
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # frame width
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # frame height

        # Prepare VideoWriter for side-by-side output (original + overlay)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            os.path.join(output_dir, "video_original.mp4"),
            fourcc,
            fps_input,
            (width * 2, height)
        )

        print(f"Processing {total_frames} frames at {fps_input} FPS ({width}×{height})…")

        frame_num = 0
        # Limit progress bar to either total_frames or self.max_fps, whichever is smaller
        pbar = tqdm(total=min(total_frames, self.max_fps), desc="Processing frames")

        while frame_num < self.max_fps and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Update tracking state for this frame
            self.process_frame(frame)              # updates self.duck_positions, self.lost_ducks, self.frame_count
            self.last_frame = frame.copy()         # store raw frame for final summary

            # 2) Quick visualization overlay with OpenCV
            vis = frame.copy()

            # 2.1) Draw lost ducks in grey
            for lost_id, data in self.lost_ducks.items():
                traj = data['trajectory']                       # list of past positions
                if len(traj) > 1:
                    for i in range(len(traj) - 1):
                        cv2.line(vis, tuple(traj[i]), tuple(traj[i+1]),
                                (128, 128, 128), 1, cv2.LINE_AA)
                last_pt = traj[-1]
                cv2.putText(vis, f"{lost_id}(lost)", last_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # 2.2) Draw active ducks
            for duck_id, info in self.duck_positions.items():
                x1, y1, x2, y2 = info['bbox']                  # bounding box
                color = (0,255,255) if info['color']=='yellow' else (0,0,0)
                # draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                # draw trajectory lines
                pts = list(self.tracked_ducks[duck_id]['positions'])
                for i in range(len(pts)-1):
                    cv2.line(vis, pts[i], pts[i+1], color, 1, cv2.LINE_AA)
                # label with ID and confidence
                label = f"{duck_id}:{info['confidence']:.2f}"
                cv2.putText(vis, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Combine original and overlay side by side
            combined = np.hstack((frame, vis))
            out.write(combined)                           # write combined frame to output video

            if display:
                cv2.imshow("Duck_Tracker", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 3) Save keyframe if requested
            if frame_num in self.keyframes:
                self.visualize_2d(frame)                   # full Matplotlib overlay
                plt.savefig(os.path.join(output_dir, f"keyframe_{frame_num:04d}.png"))
                plt.close()

            frame_num += 1
            pbar.update(1)                                # advance progress bar

        # Cleanup video IO and GUI windows
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 4) Final summary with Matplotlib if enabled
        if self.mat_summary and self.last_frame is not None:
            self.visualize_2d(self.last_frame)
            plt.savefig(os.path.join(output_dir, "final_frame_2d.png"))
            plt.close()
            self.visualize_trajectories(os.path.join(output_dir, "duck_trajectories.png"))

        # 5) Export all tracking data to JSON
        self.save_all_data(os.path.join(output_dir, "tracking_data.json"))

        print("Done!")


    #----------------------------------------------------------------------
    # FUNCTION PROCESS_IMAGE_SEQUENCE
    #----------------------------------------------------------------------

    def process_image_sequence(self, image_dir: str, output_dir: str = "output") -> None:
        """
        Process a sequence of images in a directory and save tracking results.

        Args:
            image_dir: Path to the directory containing input images.
            output_dir: Path where output images and data will be stored.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Gather all JPG and PNG files in the input directory
        image_files = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )

        # Iterate through up to max_fps images
        for frame_number, image_path in enumerate(image_files[:self.max_fps]):
            print(f"Processing image {frame_number+1}/{min(len(image_files), self.max_fps)}")

            # Load current image frame
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not read image: {image_path}")
                continue

            # Update tracker state with current frame
            self.process_frame(frame)                    # updates self.duck_positions, self.lost_ducks, self.frame_count

            # Record tracking data for this frame
            self.all_frames_data[frame_number] = {       # self.all_frames_data collects per-frame results
                'positions': self.duck_positions.copy(), # snapshot of current duck_positions
                'timestamp': datetime.now().isoformat(), # record processing time
                'image_path': image_path                 # original image file path
            }

            # Generate and save Matplotlib visualization of this frame
            self.visualize_2d(frame)                     # full 2D overlay with trajectories and predictions
            plt.savefig(os.path.join(output_dir,
                        f"frame_{frame_number:04d}.png"))
            plt.close()

        # After processing all images, save aggregated JSON data
        self.save_all_data(os.path.join(output_dir, "tracking_data.json"))  # dumps self.all_frames_data and params


    #----------------------------------------------------------------------
    # FUNCTION SAVE_ALL_DATA
    #----------------------------------------------------------------------

    def save_all_data(self, filename: str) -> None:
        """
        Save all tracking data to a JSON file.

        Args:
            filename: Path to the JSON file where data will be saved.
        """
        data_to_save = {
            'total_frames': len(self.all_frames_data),           # number of frames processed, from self.all_frames_data
            'grid_size':    self.grid_size,                      # (cols, rows) of the spatial grid, from self.grid_size
            'frames':       self.all_frames_data,                 # detailed per-frame tracking data
            'tracking_params': {
                'min_detection':     self.min_detection,         # minimum confidence threshold for detections
                'min_lenght_pixels': self.min_lenght_pixels      # minimum pixel separation between detections
            }
        }

        # Write the assembled tracking data dictionary to disk as JSON
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    #----------------------------------------------------------------------
    # FUNCTION VISUALIZE_TRAJECTORIES
    #----------------------------------------------------------------------

    def visualize_trajectories(self, output_path: str = None) -> None:
        """
        Generate a separate visualization of all duck trajectories.

        Args:
            output_path: File path to save the plot. If None, display it on screen.
        """
        plt.figure(figsize=(12, 8))                               # Create a new figure

        plt.style.use('dark_background')                          # Use dark background for visibility
        plt.grid(True, alpha=0.2)                                 # Show grid with low opacity

        # Draw trajectories for active ducks
        for duck_id, track_data in self.tracked_ducks.items():    # Iterate over currently tracked ducks
            if len(track_data['trajectory']) > 1:
                trajectory = np.array(track_data['trajectory'])   # Positions history for this duck
                color_info = self.color_map[track_data['color']]  # Color mapping for plot/text

                # Draw line segments with a fade-in effect
                segments = np.array([[trajectory[i], trajectory[i+1]] 
                                    for i in range(len(trajectory)-1)])
                for i, segment in enumerate(segments):
                    alpha = (i + 1) / len(segments)               # Increase opacity for newer segments
                    plt.plot(segment[:, 0], segment[:, 1], 
                            '-', color=color_info['rgb'], alpha=alpha, linewidth=2)

                # Mark start and end of the trajectory
                start_pt, end_pt = trajectory[0], trajectory[-1]
                plt.scatter(start_pt[0], start_pt[1], 
                            c=[color_info['rgb']], marker='o', s=100, 
                            label=f'{duck_id} (start)', alpha=0.5)
                plt.scatter(end_pt[0], end_pt[1], 
                            c=[color_info['rgb']], marker='*', s=200, 
                            label=f'{duck_id} (end)', alpha=0.9)

                # Draw direction arrow from penultimate to last point
                dx = end_pt[0] - trajectory[-2, 0]
                dy = end_pt[1] - trajectory[-2, 1]
                plt.arrow(end_pt[0], end_pt[1], dx, dy,
                        color=color_info['rgb'], width=0.5,
                        head_width=10, head_length=10, alpha=0.7)

        # Draw trajectories for lost ducks
        for lost_id, lost_data in self.lost_ducks.items():        # Iterate over ducks that have timed out
            if len(lost_data['trajectory']) > 1:
                lost_traj = np.array(lost_data['trajectory'])
                plt.plot(lost_traj[:, 0], lost_traj[:, 1], 
                        '--', color='gray', alpha=0.3, linewidth=1,
                        label=f'{lost_id} (lost)')

        plt.title('Duck Trajectories')                            # Plot title
        plt.xlabel('X Position')                                  # X-axis label
        plt.ylabel('Y Position')                                  # Y-axis label

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')    # Place legend outside plot
        plt.tight_layout()                                        # Adjust layout to fit legend

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save to file if path provided
            plt.close()                                           # Close figure to free memory
        else:
            plt.show()                                            # Otherwise, display on screen


#----------------------------------------------------------------------
# FUNCTION MAIN
#----------------------------------------------------------------------

def main():
    """
    Entry point for the duck tracking application.
    1) Loads the YOLOv8 model
    2) Prompts for a video file path (with a default)
    3) Processes the video to track ducks
    4) Saves the tracking data and a final trajectory plot
    """
    # 1) Path to YOLOv8 model weights
    model_adress = r'/home/strpicket/Duck_vision/yolo.pt'
    
    # 2) Instantiate the tracker with configuration parameters
    tracker = Duck_Tracker(
        model_adress,             # Path to the YOLOv8 weights file
        min_detection = 0.3,        # Minimum confidence to accept a detection
        min_lenght_pixels = 20,     # Minimum pixel separation between detections
        max_fps = 2083,             # Maximum number of frames to process
        max_ducks = 7,              # Maximum ducks to track simultaneously
        last_positions = 30,        # History length for each track
        max_pixels_lane = 50,       # Pixel distance threshold for association
        fps_tracking = 20,          # Frames without detection before marking lost
        lane_threshold = 0.3,       # Re-identification similarity threshold
        mat_summary = True,         # Whether to generate a final Matplotlib summary
        keyframes = None            # Optional list of keyframe indices
    )

    # 3) Prompt the user for a video path, fallback to default if empty
    video_path = input("Enter video path (or press Enter for default './DuckVideo.mp4'): ").strip()
    if not video_path:
        video_path = r'/home/strpicket/Duck_vision/video_original.mp4'

    # 4) Verify existence and process the video
    if os.path.exists(video_path):
        tracker.process_video(
            video_path,             # Path to input video
            output_dir="output_video3",  # Directory to save outputs
            save_frames=True,            # Save individual processed frames
            display=True,                # Display video during processing
            save_originals=True          # Save original frames alongside
        )
    else:
        print(f"Error: Video not found at {video_path}")
        print("Please ensure the file exists and the path is correct.")
        return

    # 5) Save the accumulated tracking data to JSON
    tracker.save_all_data('tracking_data.json')  # Filename for tracking data

    # 6) Generate and save the trajectory visualization
    tracker.visualize_trajectories('duck_trajectories.png')
    print("Trajectory visualization saved to 'duck_trajectories.png'")

if __name__ == "__main__":
    main()