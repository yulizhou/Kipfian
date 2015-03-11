from flask import Flask
from flask import request
from flask import render_template
# import pickle
import graphlab as gl


# recommender = gl.load_model('themodel')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/show_recommendations', methods=['GET', 'POST'])
def show_recommendations():
    lender_id = str(request.form['user_input'])

    # get model
    # rec = pickle.load(open('data/my_model.pkl', 'rb'))

    # recommend
    recommendation = recommender.recommend(lender_id, k=20)
    return render_template('show_recommendations.html', recommendation=recommendation)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
