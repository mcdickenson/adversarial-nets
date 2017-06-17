import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import IMAGE_PATH_FORMAT, PERTURBED_PATH_FORMAT, PERTURBATION_NAMES
from PIL import Image
from scipy import io

image_paths = [IMAGE_PATH_FORMAT.format(i) for i in range(12)]
perturbation_names = PERTURBATION_NAMES

# load images
images = [plt.imread(img) for img in image_paths]

# apply perturbations
for perturbation_name in perturbation_names:
  # load perturbation
  pth = 'data/mat/{}.mat'.format(perturbation_name)
  mat = io.loadmat(pth)
  perturbation = np.array(mat['r'])

  # add the perturbation to each image
  for ix, img in enumerate(images):
    # clip to ensure values are in the permitted range
    perturbed = np.clip(img + perturbation, 0, 255)

    outfile = PERTURBED_PATH_FORMAT.format(ix, perturbation_name)
    img = Image.fromarray(perturbed.astype(np.uint8))
    img.save(outfile)
