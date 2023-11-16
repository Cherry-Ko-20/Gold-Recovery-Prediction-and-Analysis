# Gold Recovery Prediction and Analysis

## Overview

This project focuses on predicting the gold recovery process in a mining industry context. The goal is to develop a predictive model that estimates the efficiency of the gold recovery process at different stages and provides insights into the concentrations of various metals.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contribution](#contributing)
- [Version History](#version-history)
- [Copy Rights](#copy-right)

## Introduction

The mining industry often involves complex processes for extracting valuable metals like gold. Predictive models can assist in optimizing these processes and improving overall efficiency. This project explores the use of machine learning techniques, specifically regression models, to predict gold recovery metrics.

## Data

The project utilizes a dataset containing various features related to the gold recovery process. The dataset includes information about input feed, concentrations of metals at different stages, and target variables such as rougher and final gold recovery. The data exploration and preprocessing steps are detailed in the [main.py](main.py) script.

## Project Structure

The project is structured as follows:

- **main.py**: The main script that includes data loading, preprocessing, feature engineering, model training, and evaluation.
- **files/**: Directory containing the input data files.
- **README.md**: Documentation file providing an overview of the project.

## Installation
> Clone this repository to your local machine. 
> https://github.com/Cherry-Ko-20/Gold-Recovery-Prediction-and-Analysis.git
> Install the required packages using the following command: (pip install -r requirements.txt)

Install the dependencies using:

>```bash
>pip install pandas, numpy, scikit-learn, matplotlib, seaborn

## Dependencies

The following Python libraries are used in this project:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage 
>The main.py script covers the entire workflow, including data preprocessing, feature engineering, model training, and evaluation. Adjust parameters, models, or features as needed for further experimentation.

## Results

This project focused on the analysis and optimization of the gold recovery process in a mining operation.
The best model was selected based on performance in cross-validation and tested using the test dataset.
In our case, the Random Forest Regressor demonstrated superior performance compared to the Linear Regressor. The lower sMAPE mean values obtained from the Random Forest model indicate its potential to deliver better predictions for the mining organization's gold recovery process. This suggests that the Random Forest model is a reliable choice for optimizing the gold recovery process in the mining operation.

## Contributing
> Contributions are welcome! If you find any issues or improvements, please submit a pull request.

## Version History
|Version|Realease Date|Year|Contributor|
|-------|-------------|----|-----------|
|Version 1.0|November 15|2023|Cherry Ko|

## Copy Rights
© [2023] [Cherry Ko]. All rights reserved.

