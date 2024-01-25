from PIL import Image
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

model = load_model('model_numtadb.h5')

model.make_predict_function()

def predict_label(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 32, 32, 1)
    predicted_class = np.argmax(model.predict(img_array))
    return predicted_class


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)