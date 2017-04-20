from general.dl import make_keras_picklable
from general.utils import VideoFrameExtractor
from PIL import Image
import moviepy
import numpy as np
from moviepy.editor import VideoFileClip

# No idea if I really need this or something was wrong with my installation...
#from moviepy.config import change_settings
#change_settings({"IMAGEMAGICK_BINARY": "/usr/local/Cellar/imagemagick/7.0.5-4/bin/convert"})



INDICATORS_CHAR_NAMES = ["bart", "homer", "lisa", "marge"]
INDICATORS_ICONS_FOLDER = "/data/simpsons/videp/characters"

make_keras_picklable()

__all__ = ['SimpsonsVideoTagger']


def _load_indicator_images(folder, grayscaled):
    imgs = []
    for n in INDICATORS_CHAR_NAMES:
        img = Image.open("{}/{}.png".format(folder, n))
        img.load()
        bg = Image.new("RGB", img.size, (0,0,0))
        bg.paste(img, mask=img.split()[3])

        if grayscaled:
            bg = bg.convert('L')

        imgs.append(np.array(bg)[None,:])

    return np.concatenate(imgs, axis=0)


class SimpsonsVideoTagger:
    def __init__(self, model, thresholds, preprocess_pipeline=None):
        self.model = model
        self.thresholds = thresholds
        self.preprocess_pipeline = preprocess_pipeline

        self.indicator_icons = [
            _load_indicator_images(INDICATORS_ICONS_FOLDER, grayscaled=True),
            _load_indicator_images(INDICATORS_ICONS_FOLDER, grayscaled=False)
        ]


    def _predict(self, X):
        if len(X.shape) == 3:
            X = X[None,:]

        if self.preprocess_pipeline is not None:
            X = self.preprocess_pipeline.transform(X)

        results = self.model.predict(X) > self.thresholds
        return results


    def _create_indicators_make_frame_funcs(self, preds, ts):
        ts = np.array(ts)

        def make_indicator_frame(char_id):
            char_preds = preds[:,char_id]
            def make_frame(t):
                # find pos in ts
                ts_pos = np.searchsorted(ts, t, side='right') - 1
                # find current prediction
                curr_pred = char_preds[ts_pos].astype(bool)
                if curr_pred:
                    ind_img = self.indicator_icons[1][char_id][:,:,:3]
                else:
                    ind_img = np.repeat(self.indicator_icons[0][char_id][:,:,None], 3, axis=-1)
                return ind_img
            return make_frame

        funcs = []

        for char_id in range(len(INDICATORS_CHAR_NAMES)):
            funcs.append(make_indicator_frame(char_id))
        return funcs


    def tag(self, input_video_filename, extractor_params={}):
        original_video = VideoFileClip(input_video_filename)

        fr_ext = VideoFrameExtractor(input_video_filename)
        frames, frames_ts = fr_ext.extract(**extractor_params)

        # Create the indicators clip (bottom)
        frames_preds = self._predict(frames)
        ind_frame_funcs = self._create_indicators_make_frame_funcs(frames_preds, frames_ts)

        ind_clips = []
        max_x = 0
        max_y = 0
        for i, func in enumerate(ind_frame_funcs):
            clip = moviepy.editor.VideoClip(func, duration=original_video.duration)
            clip.fps = max(int(frames.shape[0] / original_video.duration), 1)
            clip = clip.set_pos((max_x,0))

            max_x += clip.size[0]
            max_y = max(max_y, clip.size[1])

            ind_clips.append(clip)

        ind_clip = moviepy.editor.CompositeVideoClip(ind_clips, size=(max_x, max_y))

        # compose the full clip
        orig_x, orig_y = original_video.size
        ind_x, ind_y = ind_clip.size

        text = moviepy.editor.TextClip("zachmoshe.com", color='white', fontsize=20)
        text_x, text_y = text.size

        clip = moviepy.editor.CompositeVideoClip([
                original_video.set_pos((0,0)),
                ind_clip.set_pos(((orig_x-ind_x)//2, orig_y)),
                text.set_pos((10, orig_y+ind_y-text_y-10))
            ],
            size=(orig_x, orig_y+ind_y))
        clip = clip.set_duration(original_video.duration)
        return clip
