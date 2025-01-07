# from asyncio import tasks
# import json
# from textwrap import dedent
# import pendulum
# import os
# from airflow import DAG
# from airflow.operators.python import PythonOperator


# with DAG(
#     'batch_prediction',
#     default_args={'retries':2},
#     description='house price prediction tool',
#     schedule_interval='@weekly',
#     start_date=pendulum.datetime(2025,1,6,tz='UTC'),
#     catchup=False,
#     tags=['mlproject'],
    
# ) as dag:
    
#     def download_files(**kwargs):
#         pass
    
#     def batch_prediction(**kwargs):
#         from 
    