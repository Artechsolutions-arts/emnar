import psycopg2

passwords = ["Artech@707", "password", "postgres", "admin"]
for p in passwords:
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password=p,
            dbname="postgres"  # try 'postgres' first as it always exists
        )
        print(f"SUCCESS: Password is '{p}'")
        conn.close()
        break
    except Exception as e:
        print(f"FAILED: Password '{p}': {e}")
