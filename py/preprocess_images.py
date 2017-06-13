import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL


# load images
PATH_TEMPLATE_ILSVRC = 'data/img/image_originalonly_{}.jpg'
PATH_TEMPLATE_MOBILE = 'data/img/natimg{}_cropped_googlenet_orig.jpg'
image_paths = [PATH_TEMPLATE_ILSVRC.format(i) for i in range(1, 9)]
image_paths += [PATH_TEMPLATE_MOBILE.format(i) for i in range(1, 5)]

# crop and scale images
for ix, pth in enumerate(image_paths):
  print ix, pth
  data = plt.imread(pth)

  # remove label text and edge buffer
  subset = data[12:460, 12:460, ]

  # resize image to match perturbation size
  img = PIL.Image.fromarray(subset)
  resized = img.resize((224, 224), PIL.Image.BICUBIC)
  resized.save('data/img/img{}.jpg'.format(ix))

