import ffmpeg
import os
import numpy as np
import cv2 as cv
import re
import hashlib


def limit_fps(src: str, fps: int, dest=None) -> None:

    """
    Limits the frame rate of a video to the specified fps

    Parameters:
        src (str): The source path of the original video.
        dest (str): The destination path to save the video.
        fps (int): The desired frame rate.

    This function uses ffmpeg to read the source video, change its frame rate and save it to the 
    destination path.
    """

    base_name, extension = os.path.splitext(src)

    if (dest is None):
        video_output = f'{base_name}-limited({fps}fps){extension}'
    else:
        video_output = dest    

    try:
        ffmpeg.input(src).output(video_output, vf=f"fps={fps}").run()
        print(f"Saved on {video_output}")
        return video_output
    except ffmpeg.Error as e:
        print("Error:", e)


def apply_NMS(boxes: list, scores:list) -> np.array:

    """
        Apply non max suppression technique to select the best bounding boxes 
        out of a set of overlapping boxes 
    """

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-Maxium Suppresion
    indices = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                              score_threshold=0.5, nms_threshold=0.4)

    return indices


def draw_text(frame, text, ptx, pty, txt_pt, bg_color, txt_color):

    """
        Draw text box
    """

    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 0.5, 1)
    text_width, text_height = text_size[0] 

    cv.rectangle(frame, (ptx, pty - text_height - 10), (ptx + text_width, pty),
                 bg_color, -1) 
    cv.putText(frame, text, txt_pt, cv.FONT_HERSHEY_COMPLEX, 0.5, txt_color)


def is_valid_license_plate(license_plate: str) -> bool:
    
    """
        Chech if a license plate is valid based on specific criteria.

        A valid license plate must be:
            - 6 characters long 
            - Contain alphanumeric characters 
    """

    license_plate = license_plate.replace(' ', '')
    return bool(re.match(r'^[a-zA-Z0-9]{6}$', license_plate))


def format_license_plate(license_plate: str) -> str:

    """
        Format a license plate to match the format: XXXX XX
    """

    license_plate = license_plate.upper()
    formatted_plate = license_plate[:4] + ' ' + license_plate[4:] 
    return formatted_plate


def generate_hash(license_plate: str, timestamp: str, latitude: str, longitude: str) -> str:

    """
        Generate a unique hash based on the parameters given
    """

    string = (f'{license_plate}{timestamp}{latitude}{longitude}')
    return hashlib.sha256(string.encode()).hexdigest()