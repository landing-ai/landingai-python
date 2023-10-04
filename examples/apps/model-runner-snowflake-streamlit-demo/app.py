import io, os, uuid, logging

from datetime import datetime

import streamlit as st
import requests
import pandas as pd

import snowflake.connector
import textwrap
import pandas as pd
from PIL import Image
from snowflake.connector.pandas_tools import write_pandas
from landingai.predict import EdgePredictor

'''
## Model Runner Snowflake Container Demo
'''
snowflake_config = {
	"user": os.environ.get('SNOWFLAKE_USER'),
	"password": os.environ.get('SNOWFLAKE_PASSWORD'),
    "database": os.environ.get('SNOWFLAKE_DATABASE'),
	"account": os.environ.get('SNOWFLAKE_ACCOUNT'),
    "schema": os.environ.get('SNOWFLAKE_SCHEMA'),
	"role": os.environ.get('SNOWFLAKE_ROLE'),
	"warehouse": os.environ.get('SNOWFLAKE_WAREHOUSE'),
}

MR_HOST = st.text_input('Model Runner hostname', os.environ.get('MODEL_RUNNER_HOST', 'localhost'))
MR_PORT = st.text_input('Model Runner port', os.environ.get('MODEL_RUNNER_PORT', '8000'))
#snowflake_config = st.text_area('Snowflake config', snowflake_config, height=100)
st.write(snowflake_config)

conn = snowflake.connector.connect(
    user=snowflake_config['user'],
    password=snowflake_config['password'],
    account=snowflake_config['account'],
    database=snowflake_config['database'],
    schema=snowflake_config['schema'],
    role=snowflake_config['role'],
    warehouse=snowflake_config['warehouse'],
    autocommit=True,
    login_timeout=300,
)

def run_query(sql, params=[]):
    cur = conn.cursor()
    try:
        sql = textwrap.dedent(sql).strip()
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        cur.close()


@st.cache_data()
def load_test_image():
    # Load Test Images
    test_images_raw = run_query("SELECT RELATIVE_PATH, LABEL FROM TUTORIAL_DB.DATA_SCHEMA.TEST_IMAGES")
    images_df = pd.DataFrame(test_images_raw, columns=['path', 'label'])
    return images_df

def get_image_url(s3_path):
    result = run_query(f"select GET_PRESIGNED_URL(@casting_dataset_stage, '{s3_path}') as URL")
    url = result[0][0]
    return str(url)

def fetch_image(url):
    r = requests.get(url)
    image = Image.open(io.BytesIO(r.content))
    return image

test_images_df = load_test_image()

# st.dataframe(test_images_df)

landing_predictor = EdgePredictor(host=MR_HOST, port=int(MR_PORT))

progress_text = "Inference in progress. Please wait."
progress_bar = st.progress(0, text=progress_text)
progress_bar.empty()

def inference():
    job_id = uuid.uuid4()
    images = test_images_df.head(10)
    count = len(images)
    image_paths = images['path']
    scores = [0] * count
    labels = [''] * count
    timestamps = [''] * count
    for index, s3_path in image_paths.items():
        predict_result = landing_predictor.predict(fetch_image(get_image_url(s3_path)))
        if not predict_result:
            logging.error(f"failed to run inference on image {s3_path}")
            continue
        labels[index] = predict_result[0].label_name
        scores[index] = predict_result[0].score
        timestamps[index] = datetime.now().isoformat()
        progress_bar.progress(1.0*index/count, text=progress_text)
    # create result dataframe
    result_df = pd.DataFrame({
        'JOB_ID': [str(job_id)]*count,
        'FILE_PATH': list(image_paths),
        'INFERENCE_CLASS': labels,
        'INFERENCE_SCORE': scores,
        'TIMESTAMP': timestamps,
    })
    st.dataframe(result_df)
    write_pandas(conn, result_df, 'INFERENCE_RESULTS')

bt = st.button('Run Inference on testing images', on_click=inference)
