from flask import Flask
from flask import request
from flask import render_template
import graphlab as gl
import pandas as pd


app = Flask(__name__)
MODEL_PATH = 'models/item_item_cf'
recommender = gl.load_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/show_recommendations', methods=['GET', 'POST'])
def show_recommendations():
    lender_id = str(request.form['user_input'])

    # recommend
    recommended_ids = recommender.recommend([lender_id], k=5)
    loans = pd.read_csv('data/loans.csv', delimiter=',')
    recommendation = []
    for r in recommended_ids:
        loan_id = r['loan_id']
        description = loans[loans['id'] == loan_id]['descriptions'].values[0]
        if not pd.isnull(description):
            img_id = loans[loans['id'] == loan_id]['image_id'].values[0]
            img_url = "http://www.kiva.org/img/s170/" + img_id + ".jpg"
            page_url = "http://www.kiva.org/lend/" + loan_id
            recommendation.append((description, img_url, page_url))
    return render_template('show_recommendations.html',
                           recommendation=recommendation[:3])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
