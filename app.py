from flask import *
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

model = pickle.load(open('mypk11.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    
    data3 = request.form['c']
    data4 = request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    arr = np.array([[data1, data2, data3, data4,data5,data6]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















