import numpy as np
import cv2
from typing import Tuple

class GazeProjection:
    def __init__(self, screen_width: int, screen_height: int, camera_fov_deg: float):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_fov_deg = camera_fov_deg
        self.camera_fov_rad = np.deg2rad(camera_fov_deg)


    def vector_to_screen(self, pitch: float, yaw: float) -> Tuple[int, int]:
        """
        Project a 3D gaze vector onto the screen coordinates.
        """
         # Convert angles to radians
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        
        # Convert pitch and yaw to a normalized 3D gaze vector
        x = np.cos(pitch_rad) * np.sin(yaw_rad)
        y = np.sin(pitch_rad)
        z = np.cos(pitch_rad) * np.cos(yaw_rad)
        vector = np.array([x, y, z])
        
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        
        # Assume the camera is at the center of the screen and z = 1 is the screen plane
        # Calculate the screen projection using similar triangles
        x_on_screen = (vector[0] / vector[2]) * (self.screen_width / 2) / np.tan(self.camera_fov_rad / 2) + (self.screen_width / 2)
        y_on_screen = (vector[1] / vector[2]) * (self.screen_height / 2) / np.tan(self.camera_fov_rad / 2) + (self.screen_height / 2)
        
        return int(x_on_screen), int(y_on_screen)

def main():
    gaze_projection = GazeProjection(1920, 1080, 60)  # Example: screen width, height, camera field of view
    pitch, yaw = 10, 15  # Example pitch and yaw values
    vector = gaze_projection.pitch_yaw_to_vector(pitch, yaw)
    x, y = gaze_projection.vector_to_screen(vector)
    print(f'Gaze is projected at screen coordinates: ({x}, {y})')

if __name__ == '__main__':
    main()
