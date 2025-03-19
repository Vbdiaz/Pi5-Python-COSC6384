import mysql.connector

DB_CONFIG ={
  "host": "mysql-344edd44-first-project1234.i.aivencloud.com",
  "user": "avnadmin",
  "password": "AVNS_tWRNiEjGh9XP6kCgrdE",
  "database": "real_time_db",
  "port": "26056",
  "ssl_ca": "data\portfoliomanager.pem"
}

def get_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None