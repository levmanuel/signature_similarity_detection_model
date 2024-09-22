# Signature Verification using Siamese Neural Networks

This project implements a signature verification system using Siamese Neural Networks. It's designed to compare pairs of signatures and determine whether they are from the same person or different individuals.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)

## Project Overview

Signature verification is a critical task in many fields, including banking, legal proceedings, and document authentication. This project uses deep learning techniques, specifically Siamese Neural Networks, to learn a similarity metric between signatures. The model can then be used to verify whether a given signature matches a known reference signature.

## Features

- Data preparation and augmentation for signature images
- Implementation of a Siamese Neural Network architecture
- Custom contrastive loss function
- Training with data augmentation and various callbacks
- Model evaluation and performance metrics
- Continued training capability for model improvement

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Pillow
- Scikit-learn


## Model Architecture

The model uses a Siamese Neural Network architecture with the following key components:

- Convolutional layers for feature extraction
- Global Average Pooling
- Dense layers for embedding
- Custom contrastive loss function

For more details, refer to the `create_base_network` function in the code.
