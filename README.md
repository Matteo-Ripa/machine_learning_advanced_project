Project Overview

This repository contains a project on image classification using ResNet50, comparing a model trained from scratch with a pre-trained version. The workflow includes synthetic dataset creation, model definition, training, hyperparameter fine-tuning, and evaluation.


1. Dataset Creation

Before running the main project notebook, the dataset must be created.

Open dataset_creation.py.

Inside the script, set your preferred output directories for:

  - The generated images
  
  - The JSON file containing metadata for each image

Run the script:
---------------------------
bash
python dataset_creation.py
---------------------------

Approximate requirements:

Dataset creation time: ~30 minutes 

Storage needed: ~170 MB

The script will produce:

  - Generated plot images in the /image folder
  
  - JSON file with the properties of each image (e.g., distribution type, parameters, support) in the /data folder


2. Model Architecture

The file my_resnet50.py contains the implementation of the ResNet50 architecture used for the from-scratch model.

Import this module in your training code or notebook to instantiate the scratch ResNet50 network.

The architecture is compatible with the dataset generated in the previous step.


3. Main Project Code

The main workflow is implemented in the Jupyter notebook:

project_code.ipynb

This notebook includes:

  - Data loading and preprocessing
  
  - Baseline training for both scratch and pre-trained ResNet50 models
  
  - Hyperparameter fine-tuning
  
  - Final experiments with different dataset sizes

Before running the notebook:

  - Open project_code.ipynb.
  
  - Update all paths/directories so that they correctly point to:

    - The images/ folder
    
    - The data/ folder containing the JSON metadata


4. Runtime Estimates

Execution time depends on your hardware (especially GPU availability), but the following rough estimates can be used as a guideline:

  - Baseline model training: < 20 minutes
  
  - Hyperparameter fine-tuning: ~1 hour
  
  - Final experiments with different dataset sizes: up to ~1 hour


Overall, running the entire project_code.ipynb end-to-end can take approximately 1–2 hours.


5. Report pdf file

The written report for the whole project.
