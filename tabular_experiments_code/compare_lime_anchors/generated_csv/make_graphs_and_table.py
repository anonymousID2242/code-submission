from __future__ import print_function
import argparse
import collections
import sys
import os
import pickle
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa
import seaborn  # noqa
import csv
import numpy




def make_graph():
    ks = range(1, (10) + 1)
    fig = plt.figure()
    plt.ylabel('Test Coverage (%)')
    plt.xlabel('Number of explanations')
    plt.ylim(0, 100)
    seaborn.set(font_scale=2)
    seaborn.set_style('white')
 
 
    reader = csv.reader(open("save_csv_2/adult_coverage.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='MSD-MAIRE non-Discrete')
    reader = csv.reader(open("save_csv_2/adult_precision.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='RP-MAIRE - non-Discrete')
    legend_fontsize = 10
    reader = csv.reader(open("save_csv_1/adult_coverage.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='MSD-MAIRE - Discrete')
    reader = csv.reader(open("save_csv_1/adult_precision.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='RP-MAIRE - Discrete')
    legend_fontsize = 10	
 

    lgd = plt.legend(loc='best', fontsize=legend_fontsize), # noqa
                    #  bbox_to_anchor=(1, 1))
    plt.savefig("nikhil.jpg")


def main():
   
    make_graph()
            #make_table(pickles,datasets,model)


if __name__ == '__main__':
    main()
