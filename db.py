import pymongo


def get_db(url="localhost", port=27017, db_name="stocks"):
    client = pymongo.MongoClient(url, port)
    db = client[db_name]
    return db
