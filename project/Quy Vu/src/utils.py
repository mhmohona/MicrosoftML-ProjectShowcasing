"""Utilities functions for the project"""
import json
import pandas as pd
import pyodbc


class DBConnection:
    """Class for connecting and executing SQL query on Azure Database

    Parameters
    ----------
    source_file : str, optional
        path to file containing connection details, by default './login_details.json'
    """

    def __init__(self, source_file='./login_details.json'):
        self.conn_str = self.get_conn_str(source_file)

    @staticmethod
    def get_conn_str(source_file='./login_details.json'):
        """Get connection string to connect to Azure SQL database

        Parameters
        ----------
        source_file : str, optional
            path to file containing connection details, by default './login_details.json'

        Returns
        -------
        str
            connection string
        """

        login_details = json.load(open(source_file))['database']

        driver = '{ODBC Driver 17 for SQL Server}'
        server = login_details['server']
        port = login_details['port']
        database = 'patient-records'
        username = login_details['username']
        password = login_details['password']

        conn_str = 'DRIVER={};SERVER={};PORT={};DATABASE={};UID={};PWD={}'.format(
            driver, server, port, database, username, password
        )

        return conn_str

    def query(self, statement):
        """Execute SQL statement

        Parameters
        ----------
        statement : str
            statement to be excuted

        Returns
        -------
        pd.DataFram
            Result of the executed statement
        """
        with pyodbc.connect(self.conn_str) as conn:
            resultant = pd.read_sql_query(statement, conn)

        return resultant


def load_data(folder='train'):
    features = pd.read_csv('./data/{}/features.csv'.format(folder))
    labels = pd.read_csv('./data/{}/labels.csv'.format(folder))

    return features, labels


def load_metadata():
    return json.load(open('./metadata.json'))


def update_metadata(updating_dict):
    metadata = load_metadata()
    metadata.update(updating_dict)
    with open('./metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
