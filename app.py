from flask import Flask,render_template,url_for,request
from tensorflow.keras.models import load_model 
import pickle
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	model = load_model('my_model.h5')
	tokenizer = pickle.load(open('my_tokenizer.pkl', 'rb'))

	if request.method == 'POST':
		input_text = request.form['message']
		data = [input_text]
		encoded_text = tokenizer.texts_to_sequences([data])[0]
		pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
		for i in (model.predict(pad_encoded)[0]).argsort()[-1:][::-1]:
			pred_word = tokenizer.index_word[i]
			#print("Following next word:",pred_word)
	return render_template('home.html',prediction = pred_word,value1=input_text)



if __name__ == '__main__':
	app.run(debug=True)