import psycopg2
from datetime import time

def get_connection():
    connection = psycopg2.connect(
        dbname="pedestrian_analysis",
        user="postgres",
        password="adm007adm007",
        host="localhost",
        port="5432"
    )
    return connection

def add_video(file_path: str, video_name: str):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO public.videos (file_path, video_name) VALUES (%s, %s) RETURNING id;",
                    (file_path, video_name)
                )
                return cur.fetchone()[0]
    finally:
        conn.close()

def add_zone(video_id: int, polygon_str: str):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO public.zones (video_id, polygon) VALUES (%s, %s) RETURNING id;",
                    (video_id, polygon_str)
                )
                return cur.fetchone()[0]
    finally:
        conn.close()

def add_pedestrian(zone_id: int, entry_time: time, exit_time: time, video_id: int):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.pedestrians (zone_id, entry_time, exit_time, video_id)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """,
                    (zone_id, entry_time, exit_time, video_id)
                )
                return cur.fetchone()[0]
    finally:
        conn.close()

def add_analysis(pedestrian_id: int, duration: float, total_video_duration: float):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.analysis (pedestrian_id, duration)
                    VALUES (%s, %s) RETURNING id;
                    """,
                    (pedestrian_id, duration)
                )
                return cur.fetchone()[0]
    finally:
        conn.close()

def add_camera(ip_address: str, camera_name: str):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO public.cameras (ip, password) VALUES (%s, %s) RETURNING id;",
                    (ip_address, camera_name)
                )
                return cur.fetchone()[0]
    finally:
        conn.close()