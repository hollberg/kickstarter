"""
models.py
database schemas for Kickstarter app
"""
# *** IMPORTS ***
from os import getenv
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

# Comment out below when migrating to Heroku
# try:
#     from .ref import DATABASE_URL
# except ImportError:
#     raise ImportError('Did not find ref.py')



# Create a DB Object
DB = SQLAlchemy()

# Create SQLAlchemy engine object
postgres_url = getenv('DATABASE_URL')
# Postgres URL uses the 'postgres' prefix rather than SQLAlchemy's
# preferred 'posgresQL'. Fix this
postgres_url = postgres_url.replace('postgres', 'postgresql')


# # Logic handles both local and Heroku environments
# try:
#     postgres_url = getenv('DATABASE_URL')
#     # Postgres URL uses the 'postgres' prefix rather than SQLAlchemy's
#     # preferred 'posgresql'. Fix this
#     postgres_url = postgres_url.replace('postgres', 'postgresql')
# except:
#     postgres_url = DATABASE_URL


engine = DB.create_engine(sa_url=postgres_url,
                          engine_opts={})


# *** DATA IMPORT/MIGRATION FUNCTIONS ***

# COMMENT OUT FOR DEPLOYMENT TO HEROKU
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
    table_name = 'test'
    csv_to_postgres(engine=engine,
                    file=r'data/Kickstarter_Merged_Data_With_Lat_Lng.csv',
                    table_name=table_name)

    # Query data from newly created/updated table
    results = engine.execute(f'SELECT * FROM {table_name} limit 5;')
    for record in results:
        print(record)
