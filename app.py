from flask import Flask
from flask import request
from flask import render_template
import pickle
import graphlab as gl


app = Flask(__name__)


@app.route('/')
def index():
    return '''
    <form action="/predict_page" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form>
    '''


@app.route('/recommend', methods=['POST'])
def recommend():
    lender_id = str(request.form['user_input'])

    # get model
    rec = pickle.load(open('data/my_model.pkl', 'rb'))

    # recommend
    recommends = rec.recommend(lender_id, k=20)

    return render_template(show_recommendations, recommends=recommends)


@app.route('/show_recommendations')
def show_recommendations():
    return render_template(show_recommendations)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
