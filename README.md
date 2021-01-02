# CMPE537
Computer Vision

Assignment 3 - Local Descriptor based Image Classification

[Moodle Page](https://moodle.boun.edu.tr/mod/assign/view.php?id=313570)

## Installation

    $ pip install -r requirements.txt

## Image Classification Pipeline

- Compute local descriptors

  - SIFT
  - HOG (own implementation)
  - Local Binary Patterns (own implementation)
  - SURF
  - ORB
  - CenSure
  - FAST
  - BRIEF
  - MSER
  - Super point
  - or others

- Find the dictionary

  - hierarchical K-means
  - spectral clustering (scikit-learn)
  - GMM (vlfeat implementation)
  - others

- Feature quantization

  - Bag of visual words (own implementation)
  - Fisher Vectors (vlfeat implementation)
  - VLAD (vlfeat implementation)

- Classification

  - Support Vector Machines
  - Nearest Neighbors
  - Random Forest
  - Adaboost
  - Multilayer Perceptron
  - or others

- Evaluation

  - Mean F1-score (macro average over all classes)
  - Per-class F1-score
  - Per-class Precision
  - Per-class Recall
  - Multi-class Confusion Matrix
