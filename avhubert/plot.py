from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

def plot_control_figures(results):
    """
    Should demonstrate the ability of the model to recognize normal monosyllabic words.
    We may have, for ba, ga, da, fa and va, the percentage of words correctly classified;
    So perhaps, 5 bars, or maybe, 5 groups of 2/3/4? bars, that would be :
    Actual syllable prediction percentage, then the percentage for the other 2 syllables 
    of some McGurk experiment this syllable is part of, and then maybe an "others" bar.
    Maybe, we could exclude da and va from this figure, and maybe have 2 bars,
    One for the actual syllabe, and one (2 for ba) for the corresponding McGurk syllables percentages (for comparaison)

    results: should be a map between a ms-word, and it's prediction. one can match on the first letter
    """
    ba_predictions = {'b':0, 'd':0, 'th':0, 'v':0, 'o':0} # a map, with keys 'b', 'd', 'v', 'o'
    ga_predictions = {'g':0, 'd':0, 'th':0, 'o':0} # keys 'g', 'd', 'o'
    fa_predictions = {'f':0, 'v':0, 'o':0} # keys 'f', 'v', 'o'

    for k in results:
        pred = results[k]
        if k[0] == 'b':
            p = pred[0:2] if pred[0:2]=='th' else pred[0]
            if p in ['b', 'd', 'th', 'v']:
                ba_predictions[p] += 1
            else:
                ba_predictions['o'] += 1
        if k[0] == 'g':
            p = pred[0:2] if pred[0:2]=='th' else pred[0]
            if p in ['g', 'th', 'd']:
                ga_predictions[p] += 1
            else:
                ga_predictions['o'] += 1
        if k[0] == 'f' or k[0:2] == 'ph':
            p = 'f' if pred[0:2] == 'ph' else pred[0]
            if p in ['f', 'v']:
                fa_predictions[p] += 1
            else:
                fa_predictions['o'] += 1


    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Define the axises labels and plot title
    #ax.set_title('Average confidence scores per syllable for each experiment')
    ax.set_xlabel('Phoneme')
    ax.set_ylabel('Prediction percentage')

    # Define bars for each type of syllables
    width = 0.2

    # ba plot
    ind_ba = np.arange(5) / 5
    ba_dist = []
    sum = 0
    for k in ba_predictions:
        sum += ba_predictions[k]
    for k in ba_predictions:
        ba_dist.append(ba_predictions[k] / sum)
    plt.bar(ind_ba, ba_dist, width, label='Ba predictions')

    # ga plot
    ind_ga = np.arange(4) / 4 + 1.5
    ga_dist = []
    sum = 0
    for k in ga_predictions:
        sum += ga_predictions[k]
    for k in ga_predictions:
        ga_dist.append(ga_predictions[k] / sum)
    plt.bar(ind_ga, ga_dist, width, label='Ga predictions')

    # fa plot
    ind_fa = np.arange(3) / 3 + 3
    fa_dist = []
    sum = 0
    for k in fa_predictions:
        sum += fa_predictions[k]
    for k in fa_predictions:
        fa_dist.append(fa_predictions[k] / sum)
    plt.bar(ind_fa, fa_dist, width, label='Fa predictions')


    ticks = list(ba_predictions.keys()) + list(ga_predictions.keys()) + list(fa_predictions.keys())
    ind = np.concatenate((ind_ba, ind_ga, ind_fa))
    plt.xticks(ind, ticks)
    plt.legend(bbox_to_anchor=(1.1, 0.5), loc='center left')
    #plt.yscale('log')
    plt.show()
    
    return ba_predictions, ga_predictions, fa_predictions


def plot_mcgurk_figures(bgd_results, bfv_results):
    """
    Should demonstrate the presence of absence of the McGurk effect.
    Basically, we could have a plot, where, for each McGurk samples, we have the percentage of 
    prediction for Visual, Auditory and McGurk syllables, and maybe a fourth bar for other.
    But other could just be inferred from the 3 other percentages.
    So, 2 * 3 bars

    bgd_results are lists of predictions for their respective mc_gurk samples
    """

    ## BGD Experiment

    # count occurences
    a, v, mg, o = 0, 0, 0, 0
    for pred in bgd_results:
        if pred[0] == 'b':
            a += 1
        elif pred[0] == 'g':
            v += 1
        elif pred[0] == 'd' or pred[0:2] == 'th':
            mg += 1
        else:
            o += 1

    # convert to ratios
    total = a + v + mg + o
    a /= total
    v /= total
    mg /= total
    o /= total


    #plot stuff
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.set_xlabel('Phoneme')
    ax.set_ylabel('Prediction percentage')

    # Define bars for each type of syllables
    width = 0.5

    plt.bar([0, 1, 2], [a, v, mg], width, label='BGD predictions')
    plt.xticks([0, 1, 2], ['A', 'V', 'MG'])
    plt.legend(bbox_to_anchor=(1.1, 0.5), loc='center left')
    #plt.yscale('log')
    plt.show()

def plot_mcgurk_relative_increase(results):
    """
    Some kind of plot that may show how much the percentage of McGurk predictions increased
    """
    ...

