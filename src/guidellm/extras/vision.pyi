import imageio.v3 as iio
from PIL import Image as _PILImage
from PIL.Image import Image as Image

__all__ = ["Image", "PILImage", "iio"]

PILImage = _PILImage
