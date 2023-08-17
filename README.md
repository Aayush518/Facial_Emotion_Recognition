# Facial Emotion Recognition

This repository contains code for a Facial Emotion Recognition project. The goal of this project is to build and train a deep learning model that can recognize emotions from facial expressions in images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Introduction

In this project, we utilize a convolutional neural network (CNN) architecture to perform facial emotion recognition. The code is implemented using PyTorch and includes the following components:

- Loading a hdb5 file containing image data and labels
- Data preprocessing and transformation
- Creating custom datasets and data loaders
- Model architecture definition
- Model training and evaluation

## Dataset

The dataset used for this project is FER2013, which contains facial expression images labeled with various emotions. The dataset is split into training and validation sets.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Aayush518/Facial_Emotion_Recognition.git
   cd Facial_Emotion_Recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install torch pandas scikit-learn torchvision numpy matplotlib Pillow torchsummary tqdm
   
   ```

## Usage

1. Place the FER2013 dataset CSV file (e.g., `fer2013_mini_XCEPTION.csv`) in the project directory.

2. Run the data preprocessing and model training code:
   ```bash
   python Train_Model.py
   ```

3. Evaluate the trained model:
   ```bash
   python Trained.py
   ```

## Results

Include information about the model's performance on the validation set and any additional insights or visualizations.

