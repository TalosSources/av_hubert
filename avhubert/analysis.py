import pprint

def aggregate_syllables(results, prefixes):
    # results is a map between syllables, and a list of predictions for this syllable
    # prefixes is a map between a syllable, and possible prefixes for this syllable

    occurences = {}

    for syllable in results:
    
        prefix_occurences = {}
        for prefix in prefixes[syllable]:
            prefix_occurences[prefix] = 0
        prefix_occurences['other'] = 0
         
        for prediction in results[syllable]:
          found = False
          for prefix in prefixes[syllable]:
             if prediction[:len(prefix)] == prefix:
                prefix_occurences[prefix] += 1
                found = True
                break
          if not found:
            prefix_occurences['other'] += 1

        occurences[syllable] = prefix_occurences

    return occurences


syllables_results = {

    'ba': 
        ['bar', 'bear', 'bear', 'the', 'but', 'but', 'bar', 'but', 'but', 'bear', 'ber', 'but', 'but', 'blah', 'but', 'but', 'the', 'bar', 'bear', 'bear', 'but', 'bear', 
        'bar', 'bar', 'bur', 'blah', 'blow', 'burn', 'bear', "i'm", 'burn', 'burn', 'bar', 'there', 'uh', 'bar', 'burn', 'ber', 'uh', 'bar', 'there', 'but', 'bar', 'blah', 
        'but', 'bar', 'blow', 'blah', 'i', 'i', 'but', 'i', "i'm", 'bar', 'uh', 'i', 'i', 'bar', 'brah', 'i', 'uh twenty', 'i', 'bar', 'i', 'blah', 'i', 'but', 'a'], 
    'ga': 
        ['gut', 'the', 'i', 'i', 'gear', 'i', 'i', 'and', 'i', 'good', 'gear', 'gar', 'i', 'i', 'i', 'i', 'gar', 'i', 'i', 'guer', 'gear', 'i', 'i', 'and', 'girl', 'i', 
        'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'gur', 'go', 'i', 'go', 'girl', 'the', 'girl', 'girl', 'there are', 'ah', 
        'ah', 'oh', 'uh', 'go', 'ah', 'gar', 'ah', "that's", 'go', 'god', 'uh', 'gar', 'ah', 'go', 'ah', 'god', 'i', 'ah'], 
    'da': 
        ['the', 'the', 'the', 'the', 'the', 'the', 'i', 'the', 'the', 'i', 'the', 'the', 'i', 'i', 'i', 'i', 'the', 'the', 'the', 'the', 'the', 'i', 'the', 'the', 'the', 
        'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 'the two', 'the', 'the', 'the', 'the', 'the', 'the', 'the', 
        'the', 'there', 'yeah', 'there', 'dah', 'there', 'ah', 'there', 'nah', 'oh', 'there', 'there', 'yeah', 'there', 'dah', 'there', 'dah', 'there', 'there', 'there'], 
    'fa': 
        ['far', 'for', 'for', 'for', 'fur', 'far', 'far', 'for', 'far', 'far', 'far', 'far', 'far', 'far', 'for', 'fur', 'far', 'far', 'far', 'far', 'fair', 'for', 'for', 
        'fair', 'frah', 'fair', 'flow', 'fur', 'flah', 'fur', 'fur', 'fur', 'fur', 'fur', 'flow', 'fur', 'fur', 'fur', 'fair', 'far', 'fair', 'fur', 'far', 'fur', 'fur', 
        'fur', 'uh', "i'm", 'far', 'frah', 'frah', 'for her', "i'm", 'far', 'frah', 'far', 'frah', 'for', 'frah', 'for', 'uh', 'uh', 'far', 'for her', 'far', 'frah'], 
    'va': 
        ['v', 'uh', "i'm", 'the', 'v', 'uh', 'uh', 'v', 'the', 'v', 'uh', 'the', 'i', 'the', 'for', "i'm", 'i', 'v', "i'm", 'vul', 'v', 'v', 'uh', 'i', 'ver', 'wow', 'ver', 
        'raw', 'rah', "i'm", 'rah', "we're", 'ver', 'vul', "i'm", 'vul', 'ver', 'ver', 'vul', 'ver', 'vul', 'ver', 'vul', 'vul', 'ver', 'vaah', "i'm", 'wrah', 'i', 'i', 
        "i'm", "i'm", 'i', 'rah', 'a', 'uh', 'rah', "i'm", 'ok', 'uh', 'rah', 'rah', 'rah', "i'm", 'wrah', 'rah'], 
    'bga': 
        ['begar', 'god', 'the', 'begar', 'begar', 'the', "i'm", 'i', 'pica', 'begar', "i'm", 'i', "i'm", "i'm", "i'm", 'ger', 'and', 'ger', 'ghar', 'bigger', 'bugah', 
        'begar', 'girl', 'begar', 'but girl', 'mcgur', 'but girl', 'begar', 'begar', 'but girl', 'begar', 'begar', 'girl', 'of a girl', 'begar', 'girl', 'begar', 'but girl', 
        'begun', 'and girl', 'my girl', 'begar', 'begar', 'begar', 'begar', 'uh', 'god', "that's", 'i', "i'm", 'oh', "i'm", 'god', "that's", 'begar', 'oh', 'the', "i'm", 'god', 
        "i'm", 'thank you', 'i', 'god']

}


mcgurk_results = {

    'ba_ga_da': 
        ['the', 'the', 'the', 'the', 'the', 'there', 'the', 'but', 'the', 'but', 'the', 'the', 'the', 'the', 'the', 
        'the', 'the', 'the', 'the', 'i', 'oh', 'go', 'go', 'bur', 'there', 'but', 'and', 'there', 'yeah', 'girl',
         'girl', 'but', 'there', 'but', 'go', 'the', 'there', 'gar', 'god', 'go', 'i', 'i', 'but', 'i', 'i', 'ah', 
         "i'm", 'i', 'i', 'ah', 'ah', 'i', "well i think we're about 20", 'i', 'blah', 'i', 'uh', 'i', 'i', "i'm"], 

    'ba_fa_va': 
        ['the', 'the', 'the', 'the', 'but', 'fur', 'the', 'for', 'the', 'for', 'i', 'the', 'the', 'but', 'for', 
        'the', 'the', 'the', 'the', 'for', 'oh', 'v', 'vul', 'ver', 'uh', 'uh', 'ver', 'ver', 'uh', 'were', 'vert', 
        'ver', "i'm", "we're", "we're", 'for', 'the', 'ver', 'the', "i'm", 'i', 'i', 'but', 'i', "i'm", 'rah', 'uh', 
        'i', 'i', 'rah', 'ah', 'the', "but i think we're about 20", 'ah', 'there', 'i', 'i', 'i', 'i', 'a'], 

    'ga_ba_bga': 
        ['gut', 'the', 'ya', 'thank you', 'gar', 'i', 'i', 'i', 'i', 'gay', 'gar', 'gear', 'i', 'gear', 'i', 'a', 
        'gear', 'gear', 'the', 'gear', 'and', 'girl', 'i', 'girl', 'girl', 'girl', 'girl', 'girl', 'girl', 'go', 
        'girl', 'go and', 'girl', 'go', 'i', 'go', 'go and', 'i', 'girl', 'girl', 'oh', 'go', 'girl', 'oh', 'god', 
        'rah', 'gar', 'gar', 'gar', "that's", 'go', 'go', 'ah', 'girl', 'ah', 'go', 'yeah', 'gar', 'i', 'ah']
        
}

syllables_prefixes = {
   'ba' : ['b', 'th'],
   'ga' : ['g', 'th'],
   'da' : ['d', 'th'],
   'fa' : ['f'],
   'va' : ['v', 'w', 'r'],
   'bga' : ['beg', 'big', 'th', 'b', 'g']
}

mcgurk_prefixes = {
   'ba_ga_da' : ['th', 'd', 'b', 'g'],
   'ba_fa_va' : ['th', 'v', 'w', 'b', 'f'],
   'ga_ba_bga' : ['beg', 'g', 'b']
}


syllable_occurences = aggregate_syllables(syllables_results, syllables_prefixes)
print(f"\n\n====================== AGGREGATE SYLLABLES RESULTS ====================\n\n")
pprint.pprint(syllable_occurences)

mcgurk_occurences = aggregate_syllables(mcgurk_results, mcgurk_prefixes)
print(f"\n\n====================== AGGREGATE MCGURK RESULTS ====================\n\n")
pprint.pprint(mcgurk_occurences)