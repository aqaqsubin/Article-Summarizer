import psycopg2 as pg2

class dbHandler:
    def __init__(self, host, dbname, user, password, port):
        self.conn = pg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)  # db에 접속
        self.cur = self.conn.cursor()

    def query(self, sql):
        result = []
        try:
            self.cur.execute(sql)
            rows = self.cur.fetchall()
            result = list(rows)
        except (Exception, pg2.DatabaseError) as error:
            print(error)
           
        return result

    def close(self):
         if self.conn is not None:
             self.conn.close()    

