import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_ACC(ground_truth,scores,threshold):
    pred = scores >= threshold
    return 100*(pred == ground_truth).mean()

def compute_FNMR(ground_truth,scores,threshold):
    pred = scores >= threshold
    return 100*np.mean(pred[ground_truth == 1] == 0)

def compute_FMR(ground_truth,scores,threshold):
    pred = scores >= threshold
    return 100*np.mean(pred[ground_truth == 0] == 1)

def compute_BACC(ground_truth,scores,threshold):
    FNMR = compute_FNMR(ground_truth,scores,threshold)
    FMR = compute_FMR(ground_truth,scores,threshold)
    return 100 - np.mean((FNMR,FMR))

def compute_EER(ground_truth, scores, threshold = None):
    FMR, TMR, _ = roc_curve(ground_truth, scores, pos_label=1)
    EER = 100*brentq(lambda x : 1. - x - interp1d(FMR, TMR)(x), 0., 1.)
    return EER

def compute_FMR100(ground_truth,scores,threshold = None): # FNMR where FMR = 1%
    FMR, TMR, _ = roc_curve(ground_truth, scores, pos_label=1)
    FNMR = 100*(1 - brentq(lambda x : 0.01 - interp1d(TMR,FMR)(x), 0., 1.))
    return FNMR

outcome_measures = {
  'acc': compute_ACC,
  'fmr': compute_FMR,
  'fnmr': compute_FNMR,
  'eer': compute_EER,
  'bac': compute_BACC,
  'fnmr@fmr1': compute_FMR100,
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
