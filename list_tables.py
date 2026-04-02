import psycopg2
import urllib.parse
password = urllib.parse.quote_plus('Artech@707')
conn=psycopg2.connect(f'postgresql://postgres:{password}@localhost:5432/ragchat')
cur=conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
for r in cur.fetchall():
    print(r[0])
cur.close()
conn.close()
