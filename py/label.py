import io
import json
import os

from google.cloud import vision

# Helper function to serialize label data
def serialize_label(label):
  serialized = {
    'label': label.description,
    'score': label.score
  }
  return serialized

# Helper function to label an image
def label_image(client, img_filename):
  # Load the image into memory
  with io.open(img_filename, 'rb') as image_file:
      content = image_file.read()
      image = client.image(content=content)

  # Perform label detection on the image file
  labels = image.detect_labels()

  data = {
    'path': img_filename,
    'labels': [serialize_label(l) for l in labels]
  }
  return data

# Instantiate a client
vision_client = vision.Client()

# Set up image paths
image_paths = []
perturbation_names = [
  'CaffeNet',
  'GoogLeNet',
  'ResNet-152',
  'VGG-16',
  'VGG-19',
  'VGG-F'
]

for ix in range(12):
  truth_path = 'data/img/img{}.jpg'.format(ix)
  image_paths.append(truth_path)
  for p in perturbation_names:
    perturbed_path = 'data/perturbed/img{}_{}.jpg'.format(ix, p)
    image_paths.append(perturbed_path)

# Label images
output = []
for image_path in image_paths:
  print 'Labeling {}'.format(image_path)
  labeled_data = label_image(vision_client, image_path)
  output.append(labeled_data)

with open('data/analysis/labels.json', 'w') as outfile:
    json.dump(output, outfile)

