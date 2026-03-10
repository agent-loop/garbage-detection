import os
import sqlite3

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    UPLOAD_FOLDER = os.path.join(basedir, 'core', 'media')

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

db_path = os.path.join(basedir, 'garbage.db.sqlite')
conn = sqlite3.connect(db_path)

conn.execute('''CREATE TABLE IF NOT EXISTS yolo
         (image TEXT,
         date DATE,
         time DATETIME,
         ADDRESS VARCHAR(50),
         mac_address VARCHAR(500),
         is_verified INT);''')




         