"""
data_prep.py
Functions to support data processing, database loading, etc.
Not part of running app, but used to prepare and configure data
"""

import pandas as pd
from flask_sqlalchemy import SQLAlchemy


# Create a DB Object
DB = SQLAlchemy()


def csv_to_postgres(engine,
                      file: str,
                      table_name: str):
    """
    Given a *.csv filepath, create a populated table in a database
    :param engine: SQLAlchemy connection/engine for the target database
    :param file: Full filepath of the *.csv file
    :param table_name: Name of the table to be created
    :return:
    """
    df = pd.read_csv(file,
                     index_col=False)
    # print(df.head())
    # Postgres columns are case-sensitive; make lowercase
    df.columns = df.columns.str.lower()
    df.rename(columns={'unnamed: 0': 'id'},
              inplace=True)

    df.to_sql(con=engine,
              name=table_name,
              if_exists='replace',
              index=False)

    return None


# *** Define Models/Tables ***
class Kickstarter(DB.Model):
    """
    Defines the "Kickstarters" table with SQLAlchemy
    """
    id = DB.Column(DB.BigInteger, primary_key=True)
    name = DB.Column(DB.String, nullable=False)
    blurb = DB.Column(DB.String, nullable=True)
    goal = DB.Column(DB.Float, nullable=True)
    campaign_duration = DB.Column(DB.Float, nullable=True)
    current_currency = DB.Column(DB.String, nullable=True)
    fx_rate = DB.Column(DB.Float, nullable=True)
    static_usd_rate = DB.Column(DB.Float, nullable=True)
    outcome = DB.Column(DB.Boolean, nullable=True)
    days_to_success = DB.Column(DB.Float, nullable=True)
    city = DB.Column(DB.String, nullable=True)
    state = DB.Column(DB.String, nullable=True)
    country = DB.Column(DB.String, nullable=True)
    category = DB.Column(DB.String, nullable=True)
    subcategory = DB.Column(DB.String, nullable=True)
    location = DB.Column(DB.String, nullable=True)
    latitude = DB.Column(DB.Float, nullable=True)
    longitude = DB.Column(DB.Float, nullable=True)

    def __repr__(self):
        return f'Kickstarter: name - {self.name}'


# Run code to populate table
if __name__ == '__main__':
    # table_name = 'LatLong'
    # csv_to_postgres(engine=engine,
    #                 file=r'data/Kickstarter_Merged_Data_With_Lat_Lng.csv',
    #                 table_name=table_name)

    # # Query data from newly created/updated table
    # results = engine.execute(f'SELECT * FROM {table_name} limit 5;')
    # for record in results:
    #     print(record)

    table_name = 'model10k'
    csv_to_postgres(engine=engine,
                    file=r'app/data/Kickstarter_Data_For_Model_10k.csv',
                    table_name=table_name)
