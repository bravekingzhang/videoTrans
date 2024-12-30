import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from typing import List, Union

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor"""
        pass

    def extract_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames

    def extract_audio(self, video_path: Union[str, Path]) -> Path:
        """
        Extract audio from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        video_path = Path(video_path)
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(stream, str(audio_path))
        ffmpeg.run(stream, overwrite_output=True)
        
        return audio_path

    def combine_video_audio(
        self,
        frames: List[np.ndarray],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        fps: int = 30
    ) -> Path:
        """
        Combine processed frames and audio into a video
        
        Args:
            frames: List of processed frames
            audio_path: Path to the audio file
            output_path: Path for the output video
            fps: Frames per second
            
        Returns:
            Path to the output video
        """
        output_path = Path(output_path)
        temp_video = output_path.parent / f"temp_{output_path.name}"
        
        # Write frames to temporary video
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            writer.write(frame)
        writer.release()
        
        # Combine video with audio
        video = ffmpeg.input(str(temp_video))
        audio = ffmpeg.input(str(audio_path))
        stream = ffmpeg.output(
            video,
            audio,
            str(output_path),
            vcodec='libx264',
            acodec='aac'
        )
        ffmpeg.run(stream, overwrite_output=True)
        
        # Clean up temporary file
        temp_video.unlink()
        
        return output_path 