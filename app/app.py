"""
app.py - main app for the Kickstarter evaluator tool
"""

# *** IMPORTS ***
from .models_2 import DB, process_record, get_lat_value, get_lng_value
# from .models import engine    # uncomment if using Heroku DB access
# try:
#     from .model_prep import process_record
# except:
#     from model_prep import process_record
from os import getenv
from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
import pandas as pd


def create_app():
    app = Flask(__name__)

    # Give our APP access to our database
    DB.init_app(app)

    @app.route('/')
    def hello_world():  # put application's code here
        return render_template('home.html')

    # @app.route('/db')
    # def load_db():
    #     """
    #     Empty and repopulate the database with tables defined in models.py
    #     :return:
    #     """
    #     DB.drop_all()
    #     DB.create_all()
    #     new_ks = Kickstarter(id=1, name='foop')
    #     DB.session.add(new_ks)
    #     ks = Kickstarter.query.all()
    #     ks_string = str(ks[0])
    #     DB.session.commit()
    #     # df = pd.read_sql('select * from test limit 5;',
    #     #                  con=engine)
    #     # DB.table('Kickstarter')
    #     return 'Database updated!' + ks_string  # + str(df.head())

    # @app.route('/predict')
    # def prediction():
    #     """
    #
    #     :return:
    #     """
    #     return str(process_record())   # process_record())

    @app.route('/base')
    def base():
        return render_template(r'base.html')

    # @app.route('/result', methods=['GET', 'POST'])
    # def result():
    #
    #     voo = 'boo'
    #     submission = {
    #         'name_and_blurb_text': request.form['title'] + request.form['subtitle'],
    #          'goal': request.form['funding_goal'],
    #          'campaign_duration': request.form['campaign_duration'],
    #          'latitude': request.form['latitude'],
    #          'longitude': request.form['longitude'],
    #          'category': request.form['category'],
    #          'subcategory': request.form['subcategory']
    #     }
    #
    #     prediction = str(process_record(submission))
    #     print(prediction)
    #
    #     return render_template('test.html',
    #                            result={'submission': submission,
    #                            'prediction': prediction})

    @app.route("/result", methods=["GET", "POST"])
    def prediction():
        """

        :return:
        """
        if request.method == "POST":
            title = request.form["title"]
            subtitle = request.form["subtitle"]
            category = request.form["category"]
            subcategory = request.form["subcategory"]
            city = request.form["city"]
            state = request.form["state"]
            country = request.form["country"]
            campaign_duration = request.form["campaign_duration"]
            funding_goal = request.form["funding_goal"]

            location_text = city + ", " + state + ", " + country
            name_and_blurb_text = title + " " + subtitle
            campaign_duration = float(campaign_duration)
            funding_goal = float(funding_goal)


            # Convert user input location into lat/lng
            latitude = get_lat_value(location_text=location_text)
            longitude = get_lng_value(location_text=location_text)

            input_features = [
                name_and_blurb_text,
                funding_goal,
                campaign_duration,
                latitude,
                longitude,
                category,
                subcategory,
            ]
            prediction = process_record(input_feature_list=input_features,
                                        model_path=r"app/data/pickle_rf.pkl")

        return render_template("result.html", prediction=prediction)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
