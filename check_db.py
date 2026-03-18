import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='Artech@707', dbname='ragchat')
cur  = conn.cursor()

cur.execute("""
    SELECT doc_id, COUNT(*) as chunks, MIN(chunk_index), MAX(chunk_index)
    FROM rag.md_chunks
    GROUP BY doc_id
    ORDER BY doc_id
""")
print("=== rag.md_chunks summary ===")
for row in cur.fetchall():
    print(f"  doc_id={row[0]}  chunks={row[1]}  index_range={row[2]}..{row[3]}")

print()
cur.execute("""
    SELECT chunk_index, LEFT(content, 100)
    FROM rag.md_chunks WHERE doc_id = 'DEC-U-2-JV-19-20-5'
    ORDER BY chunk_index
""")
print("=== DEC-U-2-JV-19-20-5 chunks ===")
for idx, preview in cur.fetchall():
    print(f"  [{idx:02d}] {preview!r}")

conn.close()
