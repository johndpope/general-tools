import glob
import numpy as np
import scipy
from moviepy.editor import VideoFileClip
import keras.preprocessing.image
import PIL
import math


import logging
log = logging.getLogger(__name__)

__all__ = ['SimpsonsFrameGenerator']


class SimpsonsFrameGenerator:
    def __init__(self, preprocess_pipeline=None,
            background_images=None, background_videoclips_path=None,
            background_required_num=None, background_output_shape=None):

        # check if preprocess is valid.
        # valid = all transformers
        self.preprocess_pipeline = preprocess_pipeline
        if self.preprocess_pipeline:
            if not all(hasattr(step[1], 'transform') for step in self.preprocess_pipeline.steps):
                raise Exception("all steps in preprocess_pipeline must implement transform")


        # Load background images
        if background_images is None:
            if any(x is None for x in [background_videoclips_path, background_required_num, background_output_shape]):
                raise Exception("if background_images is not given, background_videoclips_path, background_required_num, background_output_shape must be presence")
            background_images = self.load_background_images(background_videoclips_path, background_required_num, background_output_shape)
        self.background_images = background_images
        self.background_shape = background_images.shape[1:]


    GENERATE_RESCALE_INTERP = 'nearest'
    def generate(  self, X,y, output_shape=(100,100), batch_size=32,
                        train_shape_range=[1.,1.],
                        aug_horizontal_flip=True,
                        aug_rotation_range=0.0,
                        aug_shift_range=0.0,
                        max_num_characters=1, num_characters_probs=None):

        # check arguments
        if max_num_characters<0:
            raise Exception("max_num_characters should be >= 0")
        if num_characters_probs is None:
            num_characters_probs = np.full((max_num_characters+1,), 1./(max_num_characters+1))
        else:
            if len(num_characters_probs) != max_num_characters+1:
                raise Exception("num_characters_probs must be of length {}".format(max_num_characters+1))


        img_augmentor = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=aug_rotation_range,
            width_shift_range=aug_shift_range,
            height_shift_range=aug_shift_range,
            horizontal_flip=aug_horizontal_flip,
            zoom_range=0.0)  # we don't zoom here. we resize the whole train image before patching

        # main generator loop
        while True:
            perm_ind = np.random.permutation(X.shape[0])
            X_perm = X[perm_ind]
            y_perm = y[perm_ind]

            # run in batches
            for i in range(math.ceil(X.shape[0]/batch_size)):
                X_batch = X_perm[i*batch_size:(i+1)*batch_size]
                y_batch = y_perm[i*batch_size:(i+1)*batch_size]

                X_batch = next(img_augmentor.flow(X_batch, shuffle=False))

                X_gen = np.zeros((batch_size,) + output_shape + (3,))
                y_gen = np.zeros((batch_size, y.shape[1]))

                available_y_labels = np.vstack({tuple(row) for row in y_batch})
                num_available_y_labels = available_y_labels.shape[0]

                # iterate over all augmented training set images and stitch them
                for j in range(batch_size):
                    bg_ind = np.random.randint(self.background_images.shape[0])

                    num_characters = np.random.choice(max_num_characters+1, p=num_characters_probs)
                    chosen_chars_inds = np.random.choice(num_available_y_labels, num_characters, replace=(num_characters > num_available_y_labels))
                    chars_ind = []
                    for ind in chosen_chars_inds:
                        chosen_char_label = available_y_labels[ind]
                        random_char_ind = np.random.choice(
                                np.argwhere(np.all(y_batch == chosen_char_label, axis=1)).flatten()
                            )
                        chars_ind.append(random_char_ind)

                    chars_ind = np.array(chars_ind, dtype=int)
                    # sort chars_inds so images with trained characters (y!=0 somewhere) will be last so
                    # we won't override them while patching multiple characters
                    chars_ind = chars_ind[y_batch[chars_ind].sum(axis=1).argsort()]

                    bg = self.background_images[bg_ind]
                    cropped_bg = self.randomly_crop_img(bg, output_shape)

                    # get the rescalaed training images we want to patch
                    training_images = []
                    for char_ind in chars_ind:
                        # get the char training image
                        train_img = X_batch[char_ind].copy()

                        train_img = self.trim_nan_edges(train_img)

                        rescale_factor = np.random.uniform(*train_shape_range)
                        train_img = self.rescale_img(train_img, rescale_factor)

                        # crop middle part if needed (if img is larger than bg)
                        img_start_h = max(0, train_img.shape[0] - output_shape[0]) // 2
                        img_start_w = max(0, train_img.shape[1] - output_shape[1]) // 2
                        train_img = train_img[img_start_h:img_start_h+output_shape[0], img_start_w:img_start_w+output_shape[1]]

                        training_images.append(train_img)
                        y_gen[j] += y_batch[char_ind]

                    # then, place them randomly on the background
                    final_img = self.patch_images_on_background(cropped_bg, training_images)

                    X_gen[j] = final_img
                    y_gen = np.clip(y_gen, 0., 1.)

                # preprocess if needed:
                if self.preprocess_pipeline:
                    X_gen = self.preprocess_pipeline.transform(X_gen)

                # generate results
                yield X_gen, y_gen


    # trims nans from the edges of the image (if the character is in the middle for example)
    # returns the image. shape will be different (cropped)
    def trim_nan_edges(self, img):
        img_char_mask = np.argwhere(~np.isnan(img).all(axis=2))
        ti_r_s, ti_r_e = img_char_mask[:,0].min(), img_char_mask[:,0].max()
        ti_c_s, ti_c_e = img_char_mask[:,1].min(), img_char_mask[:,1].max()
        return img[ti_r_s:ti_r_e, ti_c_s:ti_c_e]


    # rescales an image that contain nans
    def rescale_img(self, img, rescale_factor):
        mask = ~np.any(np.isnan(img), axis=-1)
        img[~mask] = 0.

        # rescale it according to train_shape_range
        img_required_shape = tuple((np.array(img.shape) * rescale_factor).astype(int))
        img =  scipy.misc.imresize(img, img_required_shape, interp=self.GENERATE_RESCALE_INTERP)/255.
        mask = scipy.misc.imresize(mask, img_required_shape, interp=self.GENERATE_RESCALE_INTERP)

        # reapply the mask
        img[mask==0] = np.nan
        return img


    # tries to optimally place the images on the background to minimize overlapping
    # just a simple hueristic...
    def patch_images_on_background(self, bg, images):
        bg = bg.copy()

        images_widths = np.array([0] + [ img.shape[1] for img in images ])
        images_bnds = np.cumsum(images_widths) / images_widths.sum() * bg.shape[1]  # calc right end of images scaled to bg size
        images_bnds = images_bnds.astype(int)

        for i in range(len(images)):
            img = images[i]

            # horizontal
            img_start = images_bnds[i]
            img_end = images_bnds[i+1]
            if img_start + img.shape[1] >= bg.shape[1]:
                img_start = bg.shape[1] - img.shape[1]
                img_end = bg.shape[1]
            c = np.random.randint(max(1, img_end - img_start - img.shape[1] + 1)) + img_start

            # vertical
            r = np.random.randint(max(1, bg.shape[0] - img.shape[0] + 1))

            # patch it on the bg
            img_bg_indices = np.all(np.isnan(img), axis=-1)
            bg[r:r+img.shape[0], c:c+img.shape[1], :][~img_bg_indices] = img[~img_bg_indices]

        return bg


    def randomly_crop_img(self, img, crop_size):
        if img.shape[0]<crop_size[0] or img.shape[1]<crop_size[1]:
            raise Exception("can't crop. crop_size is larger than image")
        r = np.random.randint(img.shape[0] - crop_size[0] + 1)
        c = np.random.randint(img.shape[1] - crop_size[1] + 1)
        crop = img[r:r+crop_size[0], c:c+crop_size[1]].copy()
        return crop





    BACKGROUND_GEN_NUM_FRAMES_PER_EPISODE = 100
    BACKGROUND_SIMPSONS_YELLOW = np.array([255,217,15])/255
    BACKGROUND_BLACK = np.array([0,0,0])
    def load_background_images(self, videos_path, num_backgrounds, output_shape):
        log.info("generating background images:")
        bg_imgs = []
        for filename in glob.glob("{}/*".format(videos_path)):
            log.info("extracting frames from {}...".format(filename))
            bg_clip = VideoFileClip(filename)
            curr_bg_imgs = np.zeros( (self.BACKGROUND_GEN_NUM_FRAMES_PER_EPISODE,)+output_shape+(3,) )
            for idx, t in enumerate(np.linspace(0,bg_clip.duration, self.BACKGROUND_GEN_NUM_FRAMES_PER_EPISODE)):
                i = scipy.misc.imresize(bg_clip.get_frame(t), output_shape) / 255.
                curr_bg_imgs[idx, :i.shape[0], :i.shape[1], :] = i
            bg_imgs.append(curr_bg_imgs)

        bg_imgs = np.concatenate(bg_imgs)
        # background is any frame with less than 0.5% 'simpsons yellow' in it and
        # less than 75% black (not the credits for example)
        # thresholds were set heuristically
        bg_imgs = bg_imgs[
                np.array([self.color_ratio(i, self.BACKGROUND_SIMPSONS_YELLOW)<0.005 for i in bg_imgs]) &
                np.array([self.color_ratio(i, self.BACKGROUND_BLACK)<0.75 for i in bg_imgs])
            ][:num_backgrounds]
        return bg_imgs

    # measures the euclidian distance of all pixels from the given color. if it's
    # less than 0.35 (heuristically set), it's considered as "a match"
    # returns the % of pixels that had a match on that color.
    def color_ratio(self, img, color):
        return (np.sqrt(np.sum(np.square(img - color), axis=2)) < 0.35).mean()
