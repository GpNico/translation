import glob
from PIL import Image
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
parser.add_argument('-n',
                    '--name',
                    type=str,
                    default='no_name',
                    dest='name',
                    help='Name of the gif.')
args = parser.parse_args()

# filepaths
fp_in = "imgs\\*.png"
fp_out = "imgs\\gifs\\{}.gif".format(args.name)

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))

img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, 
         format='GIF', 
         append_images=imgs,
         save_all=True, 
         duration=200, 
         loop=0)