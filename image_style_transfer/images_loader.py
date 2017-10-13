from PIL import Image, ImageOps
import numpy as np
import os.path
import glob


DEFAULT_STYLES_DIR =  "/styles"
DEFAULT_CONTENT_DIR =  "/content"
DEFAULT_OUTPUT_DIR = "/outputs"
DEFAULT_OUTPUT_SHAPE = (600,800)


def _get_resized_shape(img_shape, desired_shape):
    img_shape = np.array(img_shape)
    desired_shape = np.array(desired_shape)
    factor = (desired_shape / img_shape).min()
    output_shape = (img_shape*factor).astype(np.int32)
    return output_shape

def _get_image(path,name):
    img = Image.open("{}/{}.jpg".format(path, name))
    return img

def _get_image_np(path, name, output_shape=None):
    img = _get_image(path, name)
    if output_shape:
        img = img.resize(output_shape[::-1])
    return np.asarray(img, dtype=np.float32)


class ImagesLoader():
    def __init__(self, data_dir, styles_dir=DEFAULT_STYLES_DIR, content_dir=DEFAULT_CONTENT_DIR,
                        output_dir=DEFAULT_OUTPUT_DIR, output_shape=DEFAULT_OUTPUT_SHAPE):
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]
        self.styles_dir = data_dir + styles_dir
        self.content_dir = data_dir + content_dir
        self.output_dir = data_dir + output_dir
        self.output_shape = output_shape

    def image(self, dir, n):
        return _get_image_np(dir, n, _get_resized_shape(self.output_shape))
    def style(self, n):
        return _get_image_np(self.styles_dir, n, _get_resized_shape(self.output_shape))
    def content(self, n):
        return _get_image_np(self.content_dir, n, _get_resized_shape(self.output_shape))
    def output(self, n):
        return _get_image_np(self.outputs_dir, n, _get_resized_shape(self.output_shape))

    def content_and_style(self, content, style):
        content_img = _get_image(self.content_dir, content)
        style_img =   _get_image(self.styles_dir, style)

        # resize content
        content_resized_shape = _get_resized_shape(content_img.size[::-1], self.output_shape)
        content_img = content_img.resize(content_resized_shape[::-1])
        content_img = np.asarray(content_img, dtype=np.float32)

        # make style fit to content resized shape
        style_img = ImageOps.fit(style_img, content_resized_shape[::-1])
        style_img = np.asarray(style_img, dtype=np.float32)

        return content_img, style_img

    def all_styles(self):
        return [ os.path.splitext(os.path.basename(x))[0] for x in glob.glob(self.styles_dir + "/*") ]
    def all_contents(self):
        return [ os.path.splitext(os.path.basename(x))[0] for x in glob.glob(self.content_dir + "/*") ]
