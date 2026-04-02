import psycopg2
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Artech@707",
        dbname="onyx_db"
    )
    print("SUCCESS: Connected to 'onyx_db'")
    conn.close()
except Exception as e:
    print(f"FAILED: Connected to 'onyx_db': {e}")
