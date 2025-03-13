import os
import sqlite3

from ._constant import Constant, print_warn_msg, print_error_msg
from ._file_manager import FileManager
from ._singleton import Singleton

__all__ = []


class EmptyClass:
    def __init__(self, info: str = "") -> None:
        self._info = info

    @classmethod
    def __bool__(cls: any) -> bool:
        return False

    @classmethod
    def __str__(cls: any) -> str:
        return ""


class DbManager:
    """
    class to manage DB operation
    """
    INSERT_SIZE = 10000
    FETCH_SIZE = 10000
    MAX_ROW_COUNT = 100000000

    @classmethod
    def create_connect_db(cls, db_path: str) -> tuple:
        """
        create and connect database
        """      
        if os.path.exists(db_path):
            FileManager.check_db_file_vaild(db_path)
        try:
            # timeout set int max to avoid database is locked error.
            conn = sqlite3.connect(db_path, timeout=2147483, check_same_thread=False)
        except sqlite3.Error as err:
            return EmptyClass("emoty conn"), EmptyClass("empty curs")
        
        try:
            curs = conn.cursor()
            os.chmod(db_path, Constant.FILE_AUTHORITY)
            return conn, curs
        except sqlite3.Error as err:
            return EmptyClass("empty conn"), EmptyClass("empty curs")

    @classmethod
    def destroy_db_connect(cls, conn: sqlite3.Connection, cur: sqlite3.Cursor):
        """
        destroy connect to db
        """
        if not conn or not cur:
            return
        try:
            cur.close()
        except sqlite3.Error as err:
            raise RuntimeError(f"Falied to close db connection cursor") from err
        
        try:
            conn.close()
        except sqlite3.Error as err:
            raise RuntimeError(f"Falied to close db connection") from err

    @classmethod
    def execute_sql(cls, conn: sqlite3.Connection, sql: str) -> bool:
        """
        execute sql
        """
        try:
            conn.cursor().execute(sql)
            conn.commit()
            return True
        except sqlite3.Error as err:
            print_error_msg("SQLite Error: %s" % " ".join(err.args))
            return False

    @classmethod
    def executemany_sql(cls, conn: sqlite3.Connection, sql: str, param: any) -> bool:
        """
        executemany sql
        """
        try:
            conn.cursor().executemany(sql, param)
            conn.commit()
            return True
        except sqlite3.Error as err:
            print_error_msg("SQLite Error: %s" % " ".join(err.args))
            return False

    @classmethod
    def judge_table_exist(cls, cur: sqlite3.Cursor, table_name: str) -> bool:
        """
        judge table if exit
        """
        try:
            sql = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?"
            cur.execute(sql, (table_name,))
            return cur.fetchone()[0]
        except sqlite3.Error as err:
            raise RuntimeError(f"Falied to judge table in db file") from err

    @classmethod
    def create_table_with_headers(cls, conn: sqlite3.Connection, cur: sqlite3.Cursor, table_name: str, headers: list) -> None:
        """
        create table
        """
        if cls.judge_table_exist(cur, table_name):
            return
        table_headers = ", ".join([f"{col[0]} {col[1]}" for col in headers])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({table_headers})"
        if not cls.execute_sql(conn, sql):
            raise RuntimeError("Failed to create table in profiler db file")

    @classmethod
    def insert_data_into_table(cls, conn: sqlite3.Connection, table_name: str, data: list) -> None:
        """
        insert data into certain table
        """
        index = 0
        if not data:
            return
        sql = "insert into {table_name} values ({value_form})".format(
            table_name=table_name, value_form="?, " * (len(data[0]) - 1) + "?")
        while index < len(data):
            if not cls.executemany_sql(conn, sql, data[index:index + cls.INSERT_SIZE]):
                raise RuntimeError("Failed to insert data into profiler db file")
            index += cls.INSERT_SIZE

    @classmethod
    def fetch_all_data(cls, cur: sqlite3.Cursor, sql: str) -> list:
        """
        fetch 1000 num of data each time to get all data
        """
        data = []
        try:
            cur.execute(sql)
            while True:
                res = cur.fetchmany(cls.FETCH_SIZE)
                data += res
                if len(data) > cls.MAX_ROW_COUNT:
                    print_warn_msg("The record counts in table exceed the limit!")
                    break
                if len(res) < cls.FETCH_SIZE:
                    break
            return data
        except sqlite3.Error as err:
            print_error_msg("SQLite Error: %s" % " ".join(err.args))
            return []
        
    @classmethod
    def fetch_one_data(cls, cur: sqlite3.Cursor, sql: str) -> list:
        """
        fetch one data
        """
        try:
            cur.execute(sql)
        except sqlite3.Error as err:
            print_error_msg("SQLite Error: %s" % " ".join(err.args))
            return []
        try:
            res = cur.fetchone()
        except sqlite3.Error as err:
            print_error_msg("SQLite Error: %s" % " ".join(err.args))
            return []
        return res


class BasicDb:
    def __init__(self) -> None:
        self.db_path = None
        self.conn = None
        self.curs = None

    def init(self, db_path: str) -> None:
        if self.db_path is None:
            self.db_path = db_path

    def create_connect_db(self) -> bool:
        if self.conn and self.curs:
            return True
        self.conn, self.curs = DbManager.create_connect_db(self.db_path)
        return True if (self.conn and self.curs) else False

    def get_db_path(self) -> str:
        return self.db_path

    def close(self) -> None:
        self.db_path = None
        DbManager.destroy_db_connect(self.conn, self.curs)

    def judge_table_exist(self, table_name: str) -> bool:
        return DbManager.judge_table_exist(self.curs, table_name)

    def create_table_with_headers(self, table_name: str, headers: list) -> None:
        DbManager.create_table_with_headers(self.conn, self.curs, table_name, headers)

    def insert_data_into_table(self, table_name: str, data: list) -> None:
        DbManager.insert_data_into_table(self.conn, table_name, data)

    def fetch_all_data(self, sql: str) -> list:
        return DbManager.fetch_all_data(self.curs, sql)

    def fetch_one_data(self, sql: str) -> list:
        return DbManager.fetch_one_data(self.curs, sql)


@Singleton
class TorchDb(BasicDb):
    def __init__(self) -> None:
        super().__init__()


@Singleton
class AnalysisDb(BasicDb):
    def __init__(self) -> None:
        super().__init__()
