#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size':12,'text.usetex':True})


##################################### DATA #####################################

def get_data(arc_mcqa_seed=False,  arc_mcqa_epoch=False,
             qqp_qpc_seed=False,   qqp_qpc_epoch=False,
             wikiqa_ar_seed=False, wikiqa_ar_epoch=False,
             conll_ner_seed=False, conll_ner_epoch=False):

    """
    GET_DATA: Retrieves the data relevant to each task.
    
    Args
        arc_mcqa_seed   (bool): Flag to retrieve data from ARC Multiple Choise QA task from seed.
        arc_mcqa_epoch  (bool): Flag to retrieve data from ARC Multiple Choise QA task from epoch.
        qqp_qpc_seed    (bool): Flag to retrieve data from QQP Question-Pair Classification task from seed.
        qqp_qpc_epoch   (bool): Flag to retrieve data from QQP Question-Pair Classification task from epoch.
        wikiqa_ar_seed  (bool): Flag to retrieve data from WikiQA Answer Reranking task from seed.
        wikiqa_ar_epoch (bool): Flag to retrieve data from WikiQA Answer Reranking task from epoch.
        conll_ner_seed  (bool): Flag to retrieve data from CoNLL Named Entity Recognition task from seed.
        conll_ner_epoch (bool): Flag to retrieve data from CoNLL Named Entity Recognition task from epoch.
    
    Returns
        dict_list (list): List of dictionaries for each task containing task
                          performances in data (ndarray), average task peformance
                          in avg (float), and the name of the task (str).

    """

    if sum([arc_mcqa_seed,  arc_mcqa_epoch,  qqp_qpc_seed,   qqp_qpc_epoch,
            wikiqa_ar_seed, wikiqa_ar_epoch, conll_ner_seed, conll_ner_epoch]) < 1:
        raise Exception('At least one task must be set to True.')

    # List storing Data from Each Task in a Dictionary
    dict_list = []


    if arc_mcqa_seed:
        task = "ARC Multiple Choice Question-Answering (From Seed)"
        avg  = 48.27188940092166
        unit = 'Accuracy (\%)'
        
        data = np.array([
             # Subgraph Size, POC, P, O, C, Oracle, Floor
             [2, 48.33141542002301,  49.942462600690455, 46.029919447640964,
                 46.95051783659379,  49.942462600690455, 46.029919447640964],
             [3, 50.63291139240506,  51.55350978135789,  49.48216340621404,
                 49.712313003452245, 52.12888377445339,  48.216340621403916],
             [4, 50.63291139240506,  50.40276179516686,  50.74798619102416,
                 50.63291139240506,  52.47410817031071,  49.712313003452245],
             [5, 52.93440736478712,  52.12888377445339,  52.013808975834294,
                 50.97813578826237,  53.16455696202531,  50.17261219792866],
             [6, 52.243958573072504, 51.32336018411968,  52.243958573072504,
                 52.12888377445339,  52.47410817031071,  50.287686996547755],
             [7, 52.243958573072504, 52.243958573072504, 52.70425776754891,
                 52.589182968929805, 52.70425776754891,  51.208285385500574],
             [8, 52.81933256616801,  52.81933256616801,  52.81933256616801,
                 52.81933256616801,  52.81933256616801,  52.81933256616801] ]).T
              
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if arc_mcqa_epoch:
        task = "ARC Multiple Choice Question-Answering (From Epoch)"
        avg  = 47.03
        unit = 'Accuracy (\%)'
        
        data = np.array([
             # Subgraph Size, POC, P, O, C, Oracle, Floor
             [2, 44.64902186421173, 47.7560414269275,   44.64902186421173,
                 44.64902186421173, 47.7560414269275,   44.64902186421173],
             [3, 47.06559263521289, 48.56156501726122,  47.06559263521289,
                 47.06559263521289, 48.56156501726122,  46.72036823935558],
             [4, 47.7560414269275,  47.87111622554661,  47.7560414269275,
                 47.7560414269275,  47.986191024165706, 46.72036823935558],
             [5, 47.7560414269275,  47.7560414269275,   47.7560414269275,
                 47.7560414269275,  47.7560414269275,   47.7560414269275] ]).T
                          
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if qqp_qpc_seed:
        task = "QQP Question-Pair Classification (From Seed)"
        avg  = 86.2537
        unit = 'Accuracy (\%)'
        
        data = np.array([
            [2, 86.11, 86.78, 84.99, 87.53, 87.53, 84.66],
            [3, 85.18, 86.33, 83.67, 86.47, 86.72, 83.67],
            [4, 88.17, 88.14, 87.78, 88.16, 88.56, 87.31],
            [5, 87.55, 87.56, 87.21, 87.70, 88.20, 86.88],
            [6, 88.49, 88.37, 88.35, 88.50, 88.81, 88.24],
            [7, 88.18, 88.45, 88.19, 88.39, 88.45, 88.18],
            [8, 88.81, 88.81, 88.81, 88.81, 88.81, 88.81] ]).T
            
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if qqp_qpc_epoch:
        task = "QQP Question-Pair Classification (From Epoch)"
        avg  = 85.86
        unit = 'Accuracy (\%)'
        
        data = np.array([
             [2, 86.03,  85.6,  85.16, 86.39, 86.42, 84.55],
             [3, 85.79,  85.75, 85.13, 86.17, 86.17, 84.13],
             [4, 86.9,   86.72, 86.48, 86.9,  86.9,  85.72],
             [5, 86.64,  86.5,  86.09, 86.54, 86.72, 85.57],
             [6, 86.82,  86.88, 86.63, 87.05, 87.05, 86.13],
             [7, 86.72,  86.54, 86.43, 86.73, 86.92, 86.05],
             [8, 86.87,  86.82, 86.65, 86.87, 86.97, 86.49],
             [9, 86.76,  86.76, 86.5,  86.62, 86.76, 86.42],
             [10, 86.89, 86.89, 86.89, 86.89, 86.89, 86.89] ]).T
            
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if wikiqa_ar_seed:
        task = "WikiQA Answer Reranking (From Seed)"
        avg  = 83.775356
        unit = 'MMR'
        
        data = np.array([
             [2, 86.2706700613969,  84.0410052910053,  84.13454270597128,
                 86.2706700613969,  88.0004409171076,  82.49986306922125],
             [3, 86.4493575207861,  86.4493575207861,  83.43228200371058,
                 88.30908289241623, 89.58774250440918, 83.43228200371058],
             [4, 87.92642983119174, 87.92642983119174, 84.43156050298907,
                 87.92642983119174, 89.58774250440916, 84.34807256235828],
             [5, 87.60015117157975, 87.23859914336106, 85.7615268329554,
                 88.45553036029227, 89.69797178130513, 84.15438397581254],
             [6, 88.91849332325523, 86.57722348198538, 85.99867724867724,
                 88.91849332325523, 89.03659611992946, 84.80473670949861],
             [7, 87.99256739732931, 87.99256739732931, 86.52777777777779,
                 85.7835726883346,  87.99256739732931, 85.45288485764677],
             [8, 86.11426051902244, 86.11426051902244, 86.11426051902244,
                 86.11426051902244, 86.11426051902244, 86.11426051902244] ]).T

        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if wikiqa_ar_epoch:
        task = "WikiQA Answer Reranking (From Epoch)"
        avg  = 86.23
        unit = 'MMR'
        
        data = np.array([
             [2, 86.11740992693375, 86.76114890400603, 86.11740992693375,
                 86.76114890400603, 87.15957763576812, 85.72058453010834],
             [3, 87.29185276804326, 86.56874871160585, 87.29185276804326,
                 87.29185276804326, 87.29185276804326, 85.78175714730335],
             [4, 86.89502737121786, 86.11740992693375, 86.89502737121786,
                 86.89502737121786, 86.89502737121786, 86.0420102384388],
             [5, 86.89502737121786, 86.89502737121786, 86.89502737121786,
                 86.89502737121786, 86.89502737121786, 86.89502737121786] ]).T
              
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if conll_ner_seed:
        task = "CoNLL Named Entity Recognition (From Seed)"
        avg  = 94.02
        unit = 'F1'

        data = np.array([
             [2, 94.18105263157895, 94.18105263157895, 93.98686205154118,
                 94.00758533501897, 94.18105263157895, 93.9085011374168],
             [3, 94.43087033448478, 94.33771486349848, 94.18105263157895,
                 94.18105263157895, 94.43087033448478, 94.1166554281861],
             [4, 94.29606538040272, 94.27921476114247, 94.26042983565107,
                 94.17712985590293, 94.37557972847627, 94.06508177373124],
             [5, 94.30305073318725, 94.33866891322663, 94.30305073318725,
                 94.41401971522453, 94.41401971522453, 94.20888476776533],
             [6, 94.32785503581962, 94.35361537165011, 94.31003961898338,
                 94.33676049216248, 94.38732513062531, 94.19996627887372],
             [7, 94.26934097421203, 94.26934097421203, 94.26934097421203,
                 94.26934097421203, 94.32689876085307, 94.23563121523681],
             [8, 94.27535620942584, 94.27535620942584, 94.27535620942584,
                 94.27535620942584, 94.27535620942584, 94.27535620942584] ]).T

        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    if conll_ner_epoch:
        task = "CoNLL Named Entity Recognition (From Epoch)"
        avg  = 94.11
        unit = 'F1'
        
        data = np.array([
             [2,  94.2466677914628,  94.2466677914628,  94.10176356425617,
                  94.2466677914628,  94.2466677914628,  94.0253164556962],
             [3,  94.20399898759807, 94.17123576549979, 94.09482031381813,
                  94.07894736842105, 94.23076923076923, 94.01046060401552],
             [4,  94.1623080816602,  94.23076923076923, 94.11169225577865,
                  94.05414523066544, 94.24763832658569, 94.01046060401552],
             [5,  94.20497680303669, 94.13947213087107, 94.11169225577865,
                  94.12955465587044, 94.20790827080347, 94.03627161535216],
             [6,  94.14642375168691, 94.14642375168691, 94.14543613969968,
                  94.0779483718576,  94.19898819561551, 94.04520917678812],
             [7,  94.16329284750337, 94.16427728116041, 94.14642375168691,
                  94.14543613969968, 94.20497680303669, 94.07994602799799],
             [8,  94.12062420919443, 94.1474110305279,  94.12062420919443,
                  94.12062420919443, 94.18016194331985, 94.08788057687443],
             [9,  94.19703103913632, 94.1474110305279,  94.13749472796289,
                  94.12955465587044, 94.19703103913632, 94.11367852926296],
             [10, 94.12062420919443, 94.12062420919443, 94.12062420919443,
                  94.12062420919443, 94.12062420919443, 94.12062420919443] ]).T
              
        dict_list.append({"task":task, "avg":avg, "data":data, "metric":unit})


    return dict_list


#################################### FORMAT ####################################

labels  = ["Number of Base Models", "POC", "P", "O", "C", "Ceiling", "Floor"]
markers = [':','*','d','s','^','--', '-.']  # avg, POC, P, O, C, Oracle, Floor


def str_to_latex(str):
    
    """
    STR_TO_LATEX: Converts String to LaTeX for Plotting.
    
    Args:
        str (str): String to be Converted.

    Returns:
        Converted LaTeX String.
    """
    
    special_char = [' ', '-']
    replace_char = [r'\,', r'}$-$\mathrm{']
    
    for idx, sc in enumerate(special_char):
        
        # Replaces Special Characters with Escaped Sequence
        if sc in str:
            str = str.replace(sc, replace_char[idx])
    
    return r"$\mathrm{{{0}}}$".format(str)


##################################### PLOT #####################################

def poc_plot(data, avg, task, metric, fname="{0}_plot.pdf"):
    
    """
    POC_PLOT: Creates a figure with 2 side-by-side plots.
    
    Args:
        data (np.ndarray):
        fname (str): Name of figure.
    
    Returns:
        None.
    """
    
    num_metrics, num_graphsize = data.shape
    num_metrics -= 1
    
    fig = plt.figure(figsize=(5,5))
    
    
    # Iterates through Performances Generated by POC, P, O, C
    for idx in range(num_metrics-1):
        
        # First value corresponds to Graph Size.
        if idx == 0:
            continue
        
        # Thicker Linewidth & Marker Size for POC Performance.
        width = 2 if idx==1 else 1
        
        # Plot POC, P, O, C Results
        plt.plot(data[0], data[idx],
                 linewidth=width,
                 marker=markers[idx],
                 markersize=5*width,
                 label=str_to_latex(labels[idx]))


    # Plots Oracle, Floor Performance
    for idx in range(num_metrics-1, num_metrics+1):
        plt.plot(data[0], data[idx],
                 linestyle=markers[idx],
                 label=str_to_latex(labels[idx]))
    
    # Highlights Region of Best & Worst Possible Performance
    plt.fill_between(data[0], data[num_metrics-1], data[num_metrics],
                     facecolor='yellow', alpha=0.4)

    # Plots the Average Performance
    plt.plot([min(data[0]), max(data[0])],  [avg, avg],
             linestyle=markers[0],
             label=str_to_latex("Average"))

    # Requires X-axis Tick Labels to be Integer Values
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add Title / Labels/ Legend.
    plt.title(str_to_latex(task))
    plt.xlabel(str_to_latex(labels[0]))
    plt.ylabel(str_to_latex(metric))
    plt.legend(loc='lower right', fontsize=10)
    
    # Saves Figure
    plt.savefig(fname.format(task.replace(' ','_')), bbox_inches='tight')
    plt.close()


################################################################################

def main():
    
    # Retrieves Data from Relevant Tasks
    tasks = get_data(arc_mcqa_seed=True,  arc_mcqa_epoch=True,
                     qqp_qpc_seed=True,   qqp_qpc_epoch=True,
                     wikiqa_ar_seed=True, wikiqa_ar_epoch=True,
                     conll_ner_seed=True, conll_ner_epoch=True)
    
    
    # Plots the Data Retrieved
    for dict in tasks:
    
        poc_plot( data   = dict.get("data"),
                  avg    = dict.get("avg"),
                  task   = dict.get("task"),
                  metric = dict.get("metric") )


################################################################################

if __name__ == '__main__':
    main()

