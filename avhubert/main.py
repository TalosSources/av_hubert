import av_hubert_mcgurk
import plot
from pprint import pprint
import cached_results

experiments = [
    ('ba', 'ga', 'da'),
    ('ba', 'fa', 'va'),
    ('ga', 'ba', 'bga')
]

syllables = ['ba', 'ga', 'da', 'fa', 'va', 'bga']

#words_control_results = av_hubert_mcgurk.predict_videos_in_dir('words/normal/simple')
words_control_results = cached_results.control_word_results
print(f"\n\n====================== WORDS CONTROL RESULTS ====================\n\n")
pprint(words_control_results)


#ba_ga_da_words_results = av_hubert_mcgurk.predict_videos_in_dir('words/mcgurk/ba_ga_da')
ba_ga_da_words_results = cached_results.ba_ga_da_words_results
#ba_fa_va_words_results = av_hubert_mcgurk.predict_videos_in_dir('words/mcgurk/ba_fa_va')
ba_fa_va_words_results = cached_results.ba_fa_va_words_results 
print(f"\n\n====================== WORDS MCGURK RESULTS ====================\n\n")
pprint(ba_ga_da_words_results)
print("\n")
pprint(ba_fa_va_words_results)



plot.plot_avhubert_figure(words_control_results, ba_fa_va_words_results.values(), ba_fa_va_words_results.values(), path='words_plot.png')


