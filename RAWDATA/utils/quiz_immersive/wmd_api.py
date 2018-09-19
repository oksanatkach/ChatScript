#!flask/bin/python
from flask import Flask, jsonify, request
import gensim

BIN_DIR = 'GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(BIN_DIR, binary=True)
word2vec_model.init_sims(replace=True)

app = Flask(__name__)

@app.route('/wmd', methods=['POST'])

def wmd():

	def compute(x, y): return word2vec_model.wmdistance(x.split(), y.split())
	
	if request.method == "POST":
		input_data = request.get_json(force=True)

		try:
			options = [op['text'] for op in input_data['options']]
			marks = [op['mark'] for op in input_data['options']]
			user_input = input_data['usertext']
			dists = [compute(x, user_input) for x in options]
			pred_l = dists.index(min(dists))
			output = marks[pred_l]
			# for the future:
			# if output < **some value**: return False
			return jsonify({'result':output})
		
		except:
			# return False # in case json is corrupted: ask for input again / ask to choose from the actual options
			return jsonify({'result':0}) # return zero for now so that there are no troubles during the demo

if __name__ == '__main__':
	app.run(debug=True)