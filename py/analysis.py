import csv
import json

from common import IMAGE_PATH_FORMAT, PERTURBED_PATH_FORMAT, PERTURBATION_NAMES

# Helper function to compute intersection-over-union
# for predicted labels
def compute_iou(truth, predicted):
  true_labels = set([t['label'] for t in truth['labels']])
  pred_labels = set([p['label'] for p in predicted['labels']])

  intersect = len(true_labels.intersection(pred_labels))
  union = len(true_labels.union(pred_labels))

  iou = intersect / float(union)
  return round(iou, 2)

# Helper function to compute intersection-over-union
# for predicted labels, weighted by score
def compute_weighted_iou(truth, predicted):
  true_scores = dict([(t['label'], t['score']) for t in truth['labels']])
  pred_scores = dict([(p['label'], p['score']) for p in predicted['labels']])

  true_labels = set(true_scores.keys())
  pred_labels = set(pred_scores.keys())

  intersect = true_labels.intersection(pred_labels)
  union = true_labels.union(pred_labels)

  intersect_score = 0
  union_score = 0
  for label in union:
    # Update union score
    if label in true_labels:
      union_score += true_scores[label]
    if label in pred_labels:
      union_score += pred_scores[label]

    # Update intersect_score
    if label in intersect:
      intersect_score += true_scores[label]
      intersect_score += pred_scores[label]

  iou = intersect_score / union_score
  return round(iou, 2)

# Load json data
json_data = open('data/analysis/labels.json').read()
input_data = json.loads(json_data)
indexed_data = dict([(j['path'], j) for j in input_data])

perturbation_names = PERTURBATION_NAMES

# Open CSV output files
with open('data/analysis/iou.csv', 'w') as iou:
  with open('data/analysis/iou_weighted.csv', 'w') as iou_weighted:
    # Initialize writers
    fieldnames = ['Image'] + perturbation_names
    iou_writer = csv.DictWriter(iou, fieldnames=fieldnames)
    iou_weighted_writer = csv.DictWriter(iou_weighted, fieldnames=fieldnames)

    # Write header rows
    iou_writer.writeheader()
    iou_weighted_writer.writeheader()

    # Process data
    for ix in range(12):
      truth_path = IMAGE_PATH_FORMAT.format(ix)
      iou_row = {'Image': ix}
      iou_weighted_row = {'Image': ix}
      for p in perturbation_names:
        perturbed_path = PERTURBED_PATH_FORMAT.format(ix, p)
        # Compute IOUs
        iou_row[p] = compute_iou(indexed_data[truth_path],
                                 indexed_data[perturbed_path])
        iou_weighted_row[p] = compute_weighted_iou(indexed_data[truth_path],
                                                indexed_data[perturbed_path])

      # Write to CSV
      iou_writer.writerow(iou_row)
      iou_weighted_writer.writerow(iou_weighted_row)
