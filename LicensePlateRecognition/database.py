import sqlite3

def get_vehicle_info():
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_info")
    data = cursor.fetchall()
    conn.close()
    return data
