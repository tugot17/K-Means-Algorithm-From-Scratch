import imageio
from os import listdir
from os.path import isfile, join

from natsort import natsort

images = []

PATH_TO_GIF = 'color-reduction.gif'
PATH_TO_IMAGES = "Images/"


filenames= [f for f in listdir(PATH_TO_IMAGES) if isfile(join(PATH_TO_IMAGES, f))]



filenames = natsort.natsorted(filenames,reverse=False)
print(filenames)




for filename in filenames:
    images.append(imageio.imread("Images/" + filename))
imageio.mimsave(PATH_TO_GIF, images, duration=1.5)