import av_hubert_mcgurk
import plot

experiments = [
    ('ba', 'ga', 'da'),
    ('ba', 'fa', 'va'),
    ('ga', 'ba', 'bga')
]

syllables = ['ba', 'ga', 'da', 'fa', 'va', 'bga']

#words_control_results = av_hubert_mcgurk.control_experiment('simple_words')
#print(f"\n\n====================== SYLLABLES CONTROL RESULTS ====================\n\n{words_control_results}\n")


#mcgurk_results = av_hubert_mcgurk.mc_gurk_experiment(experiments)
#print(f"\n\n====================== MCGURK RESULTS ====================\n\n{mcgurk_results}\n")


#mcgurk_word_results = av_hubert_mcgurk.mc_gurk_word_experiment('words/mcgurk/ba_ga_da')
#print(f"\n\n====================== MCGURK WORD RESULTS ====================\n\n{mcgurk_word_results}\n")

mcgurk_word_results = {
    'bag_gag_dag_jad_0': 'bag', 
    'bag_gag_dag_jad_1': 'the', 
    'bay_gay_day_ismail_0': 'day day they', 
    'bay_gay_day_jad_0': 'the', 
    'bay_gay_day_jad_1': 'the', 
    'beer_gear_dear_ismail_0': 'dear dear dear', 
    'beer_gear_dear_jad_0': 'beer', 
    'beer_gear_dear_jad_1': 'beer', 
    'big_gig_dig_jad_0': 'big', 
    'big_gig_dig_jad_1': 'big', 
    'bot_got_dot_ismail_0': 'but but', 
    'bot_got_dot_jad_0': 'but', 
    'bot_got_dot_jad_1': 'but', 
    'buy_guy_die_ismail_0': 'by by', 
    'buy_guy_die_jad_0': 'by', 
    'buy_guy_die_jad_1': 'i'
}


control_word_results = {
'bag_jad_0': 'bag', 
'bag_jad_1': 'bag', 
'ban_jad_0': 'ban', 
'ban_jad_1': 'but', 
'bar_ismail_0': 'bar bar bar', 
'bar_jad_0': 'you', 
'bar_jad_1': 'a', 
'base_ismail_0': 'bays bays bays', 
'base_jad_0': 'bays', 
'base_jad_1': 'and', 
'bat_ismail_0': 'but but', 
'bat_jad_0': 'but', 
'bat_jad_1': 'birth', 
'bay_ismail_0': 'bay they', 
'bay_jad_0': 'bay', 
'bay_jad_1': 'bay', 
'beer_ismail_0': 'beer beer beer', 
'beer_jad_0': 'beer', 
'beer_jad_1': 'beer', 
'berry_jad_0': "there's", 
'berry_jad_1': 'barry', 
'big_jad_0': 'big', 
'big_jad_1': 'big', 
'bot_ismail_0': 'but but', 
'bot_jad_0': 'but', 
'bot_jad_1': 'but', 
'buy_ismail_0': 'buy by', 
'buy_jad_0': 'by', 
'buy_jad_1': 'by', 
'dag_jad_0': 'dag', 
'dag_jad_1': 'dag', 
'day_ismail_0': 'day day day', 
'day_jad_0': 'day', 
'day_jad_1': 'they', 
'dear_ismail_0': 'dear dear dear', 
'dear_jad_0': 'dear', 
'dear_jad_1': 'dear', 
'die_ismail_0': 'die die die', 
'die_jad_0': 'i', 
'die_jad_1': 'i', 
'dig_jad_0': 'the', 
'dig_jad_1': 'i', 
'dot_ismail_0': 'dot dot dot', 
'dot_jad_0': 'dot', 
'dot_jad_1': 'dot', 
'fan_jad_0': 'fern', 
'fan_jad_1': 'fone', 
'far_ismail_0': 'far far far', 
'far_jad_0': 'for', 
'far_jad_1': 'for', 
'fat_ismail_0': 'but', 
'fat_jad_0': 'ferth', 
'fat_jad_1': 'ferth', 
'ferry_jad_0': 'fairy', 
'ferry_jad_1': 'ferry', 
'gag_jad_0': 'gag', 
'gag_jad_1': 'i', 
'gay_ismail_0': 'ok ok', 
'gay_jad_0': 'a', 
'gay_jad_1': 'a', 
'gear_ismail_0': 'gear gear', 
'gear_jad_0': 'year', 
'gear_jad_1': 'dear', 
'gig_jad_0': 'gig', 
'gig_jad_1': 'gig', 
'got_ismail_0': 'god god', 
'got_jad_0': 'god', 
'got_jad_1': 'god', 
'guy_ismail_0': 'guy guy guy', 
'guy_jad_0': 'i', 
'guy_jad_1': "i'm", 
'phase_ismail_0': 'bays bays bays', 'phase_jad_0': 'phase', 'phase_jad_1': 'and', 'van_jad_0': 'i', 'van_jad_1': "i'm", 'var_ismail_0': 'v var var', 'var_jad_0': 'i', 'var_jad_1': 'for', 'vase_ismail_0': 'the days days days', 'vase_jad_0': 'ways', 'vase_jad_1': 'veys', 'vat_ismail_0': "that that's that", 'vat_jad_0': 'vert', 'vat_jad_1': 'a', 'very_jad_0': 'very', 'very_jad_1': 'very'}

plot.plot_control_figures(control_word_results)
plot.plot_mcgurk_figures(list(mcgurk_word_results.values()), None)

