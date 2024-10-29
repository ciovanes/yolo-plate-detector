import ffmpeg
import os

def limit_fps(src: str, dest: str, fps: int):

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