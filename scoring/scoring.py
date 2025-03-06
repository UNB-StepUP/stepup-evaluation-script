import numpy as np


def compute_acc(ground_truth, predictions, threshold):
  pass
def compute_fmr(ground_truth, predictions, threshold):
  pass
def compute_fnmr(ground_truth, predictions, threshold):
  pass
def compute_eer(ground_truth, predictions, threshold):
  pass
def compute_bac(ground_truth, predictions, threshold):
  pass
def compute_fnmr_fmr1(ground_truth, predictions, threshold):
  pass

outcome_measures = {
  'acc': compute_acc,
  'fmr': compute_fmr,
  'fnmr': compute_fnmr,
  'eer': compute_eer,
  'bac': compute_bac,
  'fnmr@fmr1': compute_fnmr_fmr1,
}

def evaluate_submission(ground_truth, predictions, threshold):
  scores_dict = {}
  for k, f in outcome_measures.items():
      scores_dict[k] = f(ground_truth, predictions, threshold)
  return scores_dict



if __name__ == '__main__':
  import json
  import pathlib

  ground_truth_filepath = pathlib.Path('/app/input/ref/ground_truth.txt')
  predictions_filepath = pathlib.Path('/app/input/res/scores.txt')
  threshold_filepath = pathlib.Path('/app/input/res/threshold.txt')
  output_filepath = pathlib.Path('/app/output/scores.json')

  '''
  for f in [ground_truth_filepath, predictions_filepath, threshold_filepath, output_filepath]:
    print(f'{f}: {f.exists()}')
  '''


  ground_truth = np.loadtxt(ground_truth_filepath)

  predictions = np.loadtxt(predictions_filepath)
  threshold = np.loadtxt(threshold_filepath)[0]

  scores_dict = evaluate_submission(ground_truth, predictions, threshold)

  with open(output_filepath, 'w') as output_file:
    json.dump(scores_dict, output_file)
