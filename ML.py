import pyodbc
import time
import os
from datetime import datetime

# =========================
# SQL SERVER CONFIGURATION
# =========================

SERVER = "YOUR_SERVER_NAME"
DATABASE = "onefinreports"

TABLE_NAME = "tbllastrefreshdaterime"

REFRESH_COLUMN = "last_refresh_date"

CHECK_INTERVAL = 60

STATE_FILE = "state.txt"


# =========================
# SQL CONNECTION
# =========================

def get_connection():

    conn = pyodbc.connect(
        f"""
        DRIVER={{ODBC Driver 17 for SQL Server}};
        SERVER={SERVER};
        DATABASE={DATABASE};
        Trusted_Connection=yes;
        """
    )

    return conn


# =========================
# GET LAST REFRESH TIME
# =========================

def get_last_refresh():

    try:

        conn = get_connection()

        cursor = conn.cursor()

        query = f"""
        SELECT TOP 1 {REFRESH_COLUMN}
        FROM {TABLE_NAME}
        ORDER BY {REFRESH_COLUMN} DESC
        """

        cursor.execute(query)

        row = cursor.fetchone()

        conn.close()

        if row:
            return str(row[0])

        return None

    except Exception as e:

        print(f"Database Error: {e}")

        return None


# =========================
# READ SAVED STATE
# =========================

def read_saved_state():

    if not os.path.exists(STATE_FILE):
        return None

    with open(STATE_FILE, "r") as file:
        return file.read().strip()


# =========================
# SAVE STATE
# =========================

def save_state(value):

    with open(STATE_FILE, "w") as file:
        file.write(value)


# =========================
# MONITOR FUNCTION
# =========================

def monitor():

    print("================================")
    print("SQL REFRESH MONITOR STARTED")
    print("================================")

    while True:

        print("\nChecking refresh status...")
        print(f"Time: {datetime.now()}")

        current_refresh = get_last_refresh()

        saved_refresh = read_saved_state()

        print(f"Current Refresh : {current_refresh}")
        print(f"Saved Refresh   : {saved_refresh}")

        # First execution
        if saved_refresh is None and current_refresh:

            save_state(current_refresh)

            print("Initial timestamp saved.")

        # New refresh detected
        elif current_refresh and current_refresh != saved_refresh:

            print("\nNEW REFRESH DETECTED!")

            save_state(current_refresh)

        else:

            print("No new refresh detected.")

        time.sleep(CHECK_INTERVAL)


# =========================
# START
# =========================

if __name__ == "__main__":

    monitor()
