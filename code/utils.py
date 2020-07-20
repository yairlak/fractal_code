
def get_verb_indexes_in_onehot_encoding(vocab, verb_initial, features):
    '''
    !!! Assumes that the features are ['sg', 'pl'] !!!
    '''
    assert features[0] == 'sg' and features[1] == 'pl'
    word2idx = vocab.word2idx

    vocab_token_strings = word2idx.keys()
    singular_verb_token_strings = [w for w in vocab_token_strings if w[0] == verb_initial and (features[0] in w)]
    plural_verb_token_strings = [w for w in vocab_token_strings if w[0] == verb_initial and (features[1] in w)]

    return [word2idx[v] for v in singular_verb_token_strings], [word2idx[v] for v in plural_verb_token_strings]



def get_agreements_center_embedding(sentence, noun_initial, verb_initial):
    '''
    sentence - (list of strings) The sentence to analyze as a list of its word strings.
    noun_initial - (str) how to identify a noun token by its FIRST letter (e.g., 'n' for 'n1', 'n2',...)
    verb_initial - (str) how to identify a verb token by its FIRST letter (e.g., 'v' for 'v1', 'v2',...)

    Returns:
    agreements - (dict) each key is an index of a noun word in the sentence (IX_noun) and its value is a
    tuple with the corresponding verb index and features (IX_verb, features_of_noun). The features are a
    list of features (e.g., ['sg', 'm']).
    IX_nouns - (list) indexes to noun words in the sentences (counting from ZERO).
    IX_verbs - (list) indexes to verb words in the sentences (counting from ZERO).
    '''
    agreements = {}
    IX_nouns = [i for (i, w) in enumerate(sentence) if w[0] == noun_initial]
    IX_verbs = [i for (i, w) in enumerate(sentence) if w[0] == verb_initial]
    assert len(IX_nouns) == len(IX_verbs) # check that there's an equal number of nouns and verbs in the sentence

    for IX_noun, IX_verb in zip(IX_nouns, reversed(IX_verbs)):
        features_of_noun = sentence[IX_noun][1+sentence[IX_noun].find('['):sentence[IX_noun].find(']')].split(',')
        features_of_noun = [f.strip() for f in features_of_noun]
        features_of_verb = sentence[IX_verb][1+sentence[IX_verb].find('['):sentence[IX_verb].find(']')].split(',')
        features_of_verb = [f.strip() for f in features_of_verb]
        assert features_of_noun == features_of_verb
        agreements[IX_noun] = (IX_verb, features_of_noun)
        agreements[IX_verb] = (IX_noun, features_of_verb)


    return agreements, IX_nouns, IX_verbs


def get_wrong_prediction(correct_word):

    correct_feature = correct_word[1+correct_word.find('['):correct_word.find(']')].strip()
    pos = correct_word[:correct_word.find('[')]
    wrong_feature = set(args.feature_values)- set([correct_feature])
    assert len(wrong_feature)==1
    wrong_word = pos + '[' + list(wrong_feature)[0] + ']'

    return wrong_word
