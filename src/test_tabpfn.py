#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_tabpfn.py - test TabPFN digit classification
author: Bill Thompson
license: GPL 3
copyright: 2023-08-28

See https://github.com/automl/TabPFN for details of TabPFN

Data files are from:
Diabetes
https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html
Ionosphere
https://archive.ics.uci.edu/dataset/52/ionosphere
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from tabpfn.scripts.decision_boundary import DecisionBoundaryDisplay
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import sys

def GetArgs() -> argparse.Namespace:
    def ParseArgs(parser: argparse.ArgumentParser):
        class Parser(argparse.ArgumentParser):
            def error(self, message: str):
                sys.stderr.write('error: %s\n' % message)
                self.print_help()
                sys.exit(2)

        parser = Parser(description ='test_tabpfn.py - test TabPFN digit classification')
        parser.add_argument('data_file',
                            type = str,
                            help = 'A CSV file file. 0/1 response variable must be in last column.')
        parser.add_argument('-o', '--out_file',
                            type = str,
                            required = False,
                            default = 'out_file.csv',
                            help = 'Test data with prediction in last columns.')
        
        return parser.parse_args()

    parser = argparse.ArgumentParser()
    args = ParseArgs(parser)

    return args


def main():
    args = GetArgs()
    data_file = args.data_file
    output_file = args.out_file

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    df = pd.read_csv(data_file)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    classifier = TabPFNClassifier(device = device, N_ensemble_configurations=4)

    start = time.time()
    classifier.fit(X_train, y_train)
    y_eval, p_eval = classifier.predict(X_test, return_winning_probability = True)
    print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(y_test, y_eval))

    out_table = pd.DataFrame(X_test.copy().astype(str))
    out_table['y_eval'] = y_eval
    out_table['probability'] = p_eval
    out_table.to_csv(output_file, index = False)

    # PLOTTING
    # from https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ#scrollTo=Bkj2F3Q72OB0
 
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the training points
    vfunc = np.vectorize(lambda x : np.where(classifier.classes_ == x)[0])
    y_train_index = vfunc(y_train)
    y_train_index = y_train_index == 0

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_index, cmap=cm_bright)

    classifier.fit(X_train[:, 0:2], y_train_index)

    DecisionBoundaryDisplay.from_estimator(
        classifier, X_train[:, 0:2], alpha=0.6, ax=ax, eps=2.0, 
                grid_resolution=25, response_method="predict_proba")
    plt.show()

if __name__ == "__main__":
    main()
