from moviepy.editor import VideoFileClip
import numpy as np
import itertools

__all__ = ['VideoFrameExtractor']

def group_list(l, group_size):
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


class VideoFrameExtractor:
    def __init__(self, video_filename, batch_size=500):
        self.video_clip = VideoFileClip(video_filename)
        self.batch_size = batch_size

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


        #num_frames = frames_ts.shape[0]
        frames_ts_batches_it = group_list(frames_ts, self.batch_size)
        for frames_ts_batch in frames_ts_batches_it:
            frames = np.zeros((frames_ts_batch.shape[0], *self.video_clip.size[::-1], 3))

            for idx, t in enumerate(frames_ts_batch):
                frames[idx] = self.video_clip.get_frame(t) / 255.

            yield frames, frames_ts_batch
        #frames = np.zeros((num_frames, *self.video_clip.size[::-1], 3))
        # for idx, t in enumerate(frames_ts):
        #     frames[idx] = self.video_clip.get_frame(t) / 255.

        #return frames, frames_ts
