import gensim
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word2vec_model.init_sims(replace=True)


def WMD(input_data, model):
    def compute(x, y): return model.wmdistance(x.split(), y.split())

    try:
        options = [op['text'] for op in input_data['options']]
        marks = [op['mark'] for op in input_data['options']]
        user_input = input_data['usertext']
        dists = [compute(x, user_input) for x in options]
        pred_l = dists.index(min(dists))
        output = marks[pred_l]
        # for the future:
        # if output < **some value**: return False
        return output

    except:
        # return False # in case json is corrupted: ask for input again / ask to choose from the actual options
        return 0 # return zero for now so that there are no troubles during the demo