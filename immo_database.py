import sqlite3
import pandas as pd
import os 
from sqlite3 import Error



def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn


def delete_all_tasks(conn):
    """
    Delete all rows in the tasks table
    :param conn: Connection to the SQLite database
    :return:
    """
    sql = 'DELETE FROM RealEstate'
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def read_all_tasks(conn):
    sql = 'SELECT * FROM RealEstate'
    cur = conn.cursor()
    cur.execute(sql)
    print(cur.fetchall())
    conn.commit()


#os.getcwd()


if __name__ == '__main__':
    db_file = '/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/RealEstate.db'
    conn = create_connection(db_file)
    delete_all_tasks(conn)