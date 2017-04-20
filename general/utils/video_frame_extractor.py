from moviepy.editor import VideoFileClip
import numpy as np

__all__ = ['VideoFrameExtractor']


class VideoFrameExtractor:
    def __init__(self, video_filename):
        self.video_clip = VideoFileClip(video_filename)



    def extract(self, every_nth_frame=None, frames_per_second=None, num_frames=None):
        if every_nth_frame is not None:
            num_frames = int(self.video_clip.fps * self.video_clip.duration)
            frames_ts = np.array(range(0, num_frames, every_nth_frame)) / self.video_clip.fps

        elif frames_per_second is not None:
            frames_ts = np.linspace(0, self.video_clip.duration, self.video_clip.duration * frames_per_second)

        elif num_frames is not None:
            frames_ts = np.linspace(0, self.video_clip.duration, num_frames)

        else:
            raise Exception("one of every_nth_frame, frames_per_second or num_frames must be given")


        num_frames = frames_ts.shape[0]
        frames = np.zeros((num_frames, *self.video_clip.size[::-1], 3))
        for idx, t in enumerate(frames_ts):
            frames[idx] = self.video_clip.get_frame(t) / 255.

        return frames, frames_ts
