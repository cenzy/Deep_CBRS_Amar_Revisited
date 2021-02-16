import logging
import os
import subprocess
import sys

if __name__ == "__main__":
  logging.basicConfig(format="%(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)
  if len(sys.argv) != 3:
    logger.error("Invalid number of parameters.")
    exit(-1)
  
  test_filename = sys.argv[1]
  predictions_path = sys.argv[2]
  
  try:
    if not os.path.isdir(predictions_path):
      raise RuntimeError("Invalid predictions path specified. Unable to run evaluator.")
      
    for root, dirs, files in os.walk(predictions_path):
      filtered_files = list(filter(lambda x: x.startswith("predictions"), files))
      print(filtered_files)
      num_filtered_files = len(filtered_files)
      print(num_filtered_files)
      if num_filtered_files == 1:
        cutoff = str(root)[root.rfind(os.sep):].split("_")[1]
        print(cutoff)
        results_filename = os.sep.join([root, "results.tsv"])
        mimir_output = subprocess.call(["java", "-jar", "experiments/binaries/mimir.jar",
                                                "-holdout",
                                                "-cutoff", cutoff,
                                                "-test", test_filename,
                                                "-predictions", os.sep.join([root, files[0]]),
                                                "-results", results_filename])
        logger.info(mimir_output)
      elif num_filtered_files > 1:
        cutoff = str(root)[root.rfind(os.sep):].split("_")[1]
        results_filename = os.sep.join([root, "results.tsv"])
        mimir_output = subprocess.call(["java", "-jar", "experiments/binaries/mimir.jar",
                                                "-cv",
                                                "-folds",
                                                str(num_filtered_files),
                                                "-cutoff", cutoff,
                                                "-test", test_filename,
                                                "-predictions", os.sep.join([root, "predictions_%s.tsv"]),
                                                "-results", results_filename])
        logger.info(mimir_output)
  except RuntimeError as e:
    logger.exception(e)
        