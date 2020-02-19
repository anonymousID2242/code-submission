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


def load_pickles(datasets, models, out_folder):
    pickles = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0.))
    for d in datasets:
        for m in models:
            path = os.path.join(out_folder, '%s-%s.pickle' % (d, m))
            pickles[d][m] = pickle.load(open(path))
    return pickles


def make_graph(pickles, dataset, model, save=None):
    lime_sub = pickles[dataset][model]['lime_pred_submodular'][2]
    anchor_sub = pickles[dataset][model]['anchor_submodular'][2]
    ks = range(1, len(lime_sub) + 1)
    
    fig = plt.figure()
    plt.ylabel('Test Coverage (%)')
    plt.xlabel('Number of explanations')
    plt.ylim(0, 100)
    seaborn.set(font_scale=2)
    seaborn.set_style('white')
    plt.plot(ks, np.array(anchor_sub) * 100, 'o-', lw=4,
             markersize=10, label='SP-Anchor')
    plt.plot(ks, np.array(lime_sub) * 100, 's--', lw=4,
             markersize=10, label='SP-LIME')
    reader = csv.reader(open("../src/adult_coverage_msd.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='MSD-MAIRE')
    reader = csv.reader(open("../src/adult_coverage_rp.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=10, label='RP-MAIRE')
    legend_fontsize = 10
    if 'anchor_random' in pickles[dataset][model]:
        anchor_random = pickles[dataset][model]['anchor_random']
        plt.plot(
            ks, np.array(anchor_random[1]) * 100,'o-',
            lw=4, markersize=10, label='RP-Anchor')

    if 'lime_pred_random' in pickles[dataset][model]:
        lime_random = pickles[dataset][model]['lime_pred_random']
        plt.plot(
            ks, np.array(lime_random[1]) * 100,'o--',
            lw=4, markersize=10, label='RP-LIME')

    lgd = plt.legend(loc='best', fontsize=legend_fontsize), # noqa
                    #  bbox_to_anchor=(1, 1))
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
    return fig




def make_graph_prec(pickles, dataset, model, save=None):
    lime_sub = pickles[dataset][model]['lime_pred_submodular'][1]
    anchor_sub = pickles[dataset][model]['anchor_submodular'][1]
    ks = range(1, len(lime_sub) + 1)
    
    fig = plt.figure()
    plt.ylabel('Test Precision (%)')
    plt.xlabel('Number of explanations')
    plt.ylim(90, 100)
    seaborn.set(font_scale=2)
    seaborn.set_style('white')
    plt.plot(ks, np.array(anchor_sub) * 100, 'o-', lw=4,
             markersize=8, label='SP-Anchor')
    plt.plot(ks, np.array(lime_sub) * 100, 's--', lw=4,
             markersize=8, label='SP-LIME')
    reader = csv.reader(open("../src/adult_precision_msd.csv", "rb"), delimiter=",")
    x = list(reader)
    result = numpy.array(x).astype("float")
    coverage1 = result*100
    coverage1 = coverage1.reshape(10,)
    plt.plot(ks,coverage1, 's--', lw=4,
             markersize=8, label='MSD-MAIRE')
    #reader = csv.reader(open("germancredit_precision_rp.csv", "rb"), delimiter=",")
    #x = list(reader)
    #result = numpy.array(x).astype("float")
    #coverage1 = result*100
    #coverage1 = coverage1.reshape(10,)
    #plt.plot(ks,coverage1, 's--', lw=4,
      #       markersize=8, label='RP-MAIRE')
    legend_fontsize = 10
    #if 'anchor_random' in pickles[dataset][model]:
    #    anchor_random = pickles[dataset][model]['anchor_random'][0]
	#print(anchor_random)
        #plt.plot(
        #    ks, np.array(anchor_random) * 100,'o-',
        #    lw=4, markersize=8, label='RP-Anchor')


    #if 'lime_pred_random' in pickles[dataset][model]:
     #   lime_random = pickles[dataset][model]['lime_pred_random'][0]
	#print(lime_random)
        #plt.plot(
         #   ks, np.array(lime_random) * 100,'o--',
          #  lw=4, markersize=8, label='RP-LIME')

    lgd = plt.legend(loc='best', fontsize=legend_fontsize), # noqa
                    #  bbox_to_anchor=(1, 1))
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
    return fig





def main():
    parser = argparse.ArgumentParser(description='Graphs')
    parser.add_argument(
        '-r', dest='results_folder',
        default='./results') # noqa
    parser.add_argument(
        '-g', dest='graphs_folder',
        default='./graphs')

    args = parser.parse_args()
    datasets = ['adult']
    models = ['nn']
    pickles = load_pickles(datasets, models, args.results_folder)
    print('')
    #tab = make_table(pickles, datasets, models)
    #print('Table:')
    #print(tab)
    for dataset in datasets:
        for model in models:
            path = os.path.join(args.graphs_folder, '%s-%s_coverage.png' %
                                (dataset, model))
            make_graph(pickles, dataset, model, save=path)
            #make_graph_prec(pickles, dataset, model, save=path)
            #make_table(pickles,datasets,model)
	for dataset in datasets:
		    for model in models:
		        path = os.path.join(args.graphs_folder, '%s-%s_precision.png' %
		                            (dataset, model))
		        #make_graph(pickles, dataset, model, save=path)
		        make_graph_prec(pickles, dataset, model, save=path)
		        #make_table(pickles,datasets,model)

if __name__ == '__main__':
    main()
