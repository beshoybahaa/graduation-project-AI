from neo4j import GraphDatabase

uri = "neo4j+s://8fef1a11.databases.neo4j.io"
username = "neo4j"
password = "g6DTWAKPPHTvJWNBEZ4vgTDSTt99ZUkE-hlyWv7-1Bg"

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        greeting = session.run("RETURN 'Connection successful!'").single()
        print(greeting[0])
except Exception as e:
    print("Connection failed:", e)
finally:
    driver.close()
