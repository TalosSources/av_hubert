from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


def word_to_phoneme(word):
    """
    Existing phonemes categories:
    /b/ -> includes [b]
    /g/ -> includes [g]
    /d/ -> includes [d, th]
    /f/ -> includes [f, ph]
    /v/ -> includes [v] (maybe w?)
    /o/ -> includes everything else

    maps a word to a phoneme using it's first few letters
    """
    prefix_map = {
        'b' : 'b',
        'g' : 'g',
        'd' : 'd',
        'th' : 'd',
        'f' : 'f',
        'ph' : 'f',
        'v' : 'v',
    }

    for prefix in prefix_map:
        if word.startswith(prefix):
            return prefix_map[prefix]
    return 'o'


def convert_word_maps_to_figure_maps(control_map, b_g_d_preds, b_f_v_preds):
    """
    * control_map is a map between control words and their predictions.
    * ba_ga_da_preds is list of predictions for ba_ga_da samples
    * same for ba_fa_va_preds

    * ba_ga_da_results will be 3 tuples of 3 floats each
    """

    b_control_bgd = np.zeros(3) # corresponding to a, v, mg pred counts for ba
    b_control_bfv = np.zeros(3)
    g_control = np.zeros(3)
    f_control = np.zeros(3)
    for word in control_map:
        word_ph, pred_ph = word_to_phoneme(word), word_to_phoneme(control_map[word])

        # biig spaghetti
        if word_ph == 'b':
            if pred_ph == 'b':
                b_control_bgd[0] += 1
                b_control_bfv[0] += 1
            if pred_ph == 'g':
                b_control_bgd[1] += 1
            if pred_ph == 'd':
                b_control_bgd[2] += 1
            if pred_ph == 'f':
                b_control_bfv[1] += 1
            if pred_ph == 'v':
                b_control_bfv[2] += 1
        if word_ph == 'g':
            if pred_ph == 'b':
                g_control[0] += 1
            if pred_ph == 'g':
                g_control[1] += 1
            if pred_ph == 'd':
                g_control[2] += 1
        if word_ph == 'f':
            if pred_ph == 'b':
                f_control[0] += 1
            if pred_ph == 'f':
                f_control[1] += 1
            if pred_ph == 'v':
                f_control[2] += 1

    b_control_bgd /= sum(b_control_bgd) # convert to percentages
    b_control_bfv /= sum(b_control_bfv)
    g_control /= sum(g_control)
    f_control /= sum(f_control)


    # now, we build the mcgurk tuples
    b_g_d_mcgurk = np.zeros(3)
    for pred in b_g_d_preds:
        pred_ph = word_to_phoneme(pred)
        if pred_ph == 'b':
            b_g_d_mcgurk[0] += 1
        if pred_ph == 'g':
            b_g_d_mcgurk[1] += 1
        if pred_ph == 'd':
            b_g_d_mcgurk[2] += 1
    
    b_f_v_mcgurk = np.zeros(3)
    for pred in b_f_v_preds:
        print(f"pred : {pred}")
        pred_ph = word_to_phoneme(pred)
        if pred_ph == 'b':
            b_f_v_mcgurk[0] += 1
        if pred_ph == 'f':
            b_f_v_mcgurk[1] += 1
        if pred_ph == 'v':
            b_f_v_mcgurk[2] += 1

    b_g_d_mcgurk /= sum(b_g_d_mcgurk)
    b_f_v_mcgurk /= sum(b_f_v_mcgurk)

    b_g_d_results = np.array([b_control_bgd, g_control, b_g_d_mcgurk])
    b_f_v_results = np.array([b_control_bgd, f_control, b_f_v_mcgurk])

    return b_g_d_results, b_f_v_results

def plot_avhubert_figure(control_map, b_g_d_preds, b_f_v_preds, path=None):
    """
    One sublot per experiment -> 2 subplots
    subplot : 3 bars (a, v, mg) for A syllable, 3 same bars for V syllable, 3 same bars for A+V combination => 9 bars
    ba_ga_da_results should be, maybe, 3 tuples of 3 floats each?
    """

    ba_ga_da_results, ba_fa_va_results = convert_word_maps_to_figure_maps(control_map, b_g_d_preds, b_f_v_preds)



    # Extract data for plotting
    data = [ba_ga_da_results, ba_fa_va_results]
    pprint(data)
    labels = ['A', 'V', 'A+V']
    colors = {'A': 'blue', 'V': 'orange', 'A+V': 'green'}
    legend_labels = ['Audio', 'Visual', 'McGurk']
    label_colors = [colors[label] for label in labels]

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5), constrained_layout=True)

    fig.supxlabel('Input Video content')
    fig.supylabel('Ratio of prediction occurences')

    width = 0.2
    spacing = 0.1
    # Plot each experiment
    for i, ax in enumerate(axes):
        bars = []
        for j, group in enumerate(data[i].T):
            bar = ax.bar(np.arange(len(group)) + (width+spacing)*j, group, width=width, color=colors[labels[j]])
            bars.append(bar)

        # Set x-axis labels
        ax.set_xticks(np.arange(len(labels)) + (spacing+width))
        ax.set_xticklabels(labels)
        for ticklabel, tickcolor in zip(ax.get_xticklabels(), label_colors):
            ticklabel.set_color(tickcolor)

        # Set plot title
        ax.set_title(f"Experiment {i+1} ({['B..+G..=D..', 'B..+F..=V..'][i]})")

    # Add legend
    plt.legend(bars, legend_labels, bbox_to_anchor=(1.05, 1.0),title='Predicted phonemes', loc='upper left')

    # Adjust layout
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    else:
        # Show the plot
        plt.show()

def plot_control_figures(results):
    """
    Should demonstrate the ability of the model to recognize normal monosyllabic words.

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


