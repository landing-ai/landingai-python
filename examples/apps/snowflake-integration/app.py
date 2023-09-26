import base64
import json
import os

import streamlit as st
import snowflake.connector
import textwrap
import pandas as pd
from PIL import Image

import aiohttp
import asyncio


tmp_dir = '/tmp/snowflake_integration_demo'
landing_upload_api_path = 'https://app.landing.ai/pictor/v1/upload'

'''
# Snowflake + Landing Integration
'''

'''
## Integration Configuration
'''
default_snowflake_config = '''{
	"user": "ZHOUYANG.ZHANG@LANDING.AI",
	"password": "MC8-***@YvgKXKj",
	"account": "rpwerko-lai-sf-partnership",
	"role": "TEST_ROLE",
	"warehouse": "TUTORIAL_WAREHOUSE"
}
'''
land_api_key = st.text_input('Landing API key', 'land_sk_lGzgoiaB8*****Y933WdznzRL')
snowflake_config = st.text_area('Snowflake config', default_snowflake_config, height=200)


conf = json.loads(snowflake_config)
conn = snowflake.connector.connect(
    user=conf['user'],
    password=conf['password'],
    account=conf['account'],
    role=conf['role'],
    warehouse=conf['warehouse'],
    autocommit=True
)



def run_query(sql, params=[]):
    cur = conn.cursor()
    print("Connection to snowflake is done successful")
    try:
        sql = textwrap.dedent(sql).strip()
        print(f"== SNOWFLAKE ==\n{sql}\n{params}")
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        cur.close()


def upload_to_landing(list_pattern):
    asyncio.run(upload_to_landing_async(list_pattern))
    print('xxxxx done uploading to landing xxxxx')

async def upload_to_landing_async(list_pattern):
    headers = {
        'apikey': land_api_key,
        'cookie': 'koa:sess=eyJhY2NvdW50Ijp7ImlkIjoiYjJkNzk5MjAtOWJlMy0xZTAwLTAwY2UtNTIzY2ZjODM5MDEwIiwiZW1haWwiOiJ6aG91eWFuZy56aGFuZ0BsYW5kaW5nLmFpIiwiZmlyc3ROYW1lIjoiWmhvdXlhbmciLCJsYXN0TmFtZSI6IlpoYW5nIiwidXNlck5hbWUiOiIxQHpob3V5YW5nLnpoYW5nQGxhbmRpbmcuYWkiLCJhY3R.mF0aW9uQ29kZSI6bnVsbCwic3RhdHVzIjoiYWN0aXZlIiwiZm9yZ2V0UGFzc3dvcmRDb2RlIjpudWxsLCJmb3JnZXRQYXNzd29yZENvZGVDcmVhdGVkQXQiOm51bGwsInplcm9BdXRoQWNjb3VudElkIjpudWxsLCJjcmVhdGVkQXQiOiIyMDIyLTA5LTIwVDE3OjIyOjMxLjk4N1oiLCJ1cGRhdGVkQXQiOiIyMDIzLTA1LTIyVDA4OjU1OjQxLjU4OFoifSwibG9naW5UeXBlIjoiZ29vZ2xlX3NzbyIsImlzQXV0aGVudGljYXRlZCI6dHJ1ZSwidXNlciI6eyJpZCI6IjA1NmU4YmYyLTY4OTctNDg1Mi1hNWFlLWNkMzA0MmQ5M2NjYiIsImFjY291bnRJZCI6ImIyZDc5OTIwLTliZTMtMWUwMC0wMGNlLTUyM2NmYzgzOTAxMCIsIm5hbWUiOiJaaG91eWFuZyIsImxhc3ROYW1lIjoiWmhhbmciLCJ1c2VybmFtZSI6IjFAemhvdXlhbmcuemhhbmdAbGFuZGluZy5haSIsInN0YXR1cyI6IkFDVElWRSIsImVtYWlsIjoiemhvdXlhbmcuemhhbmdAbGFuZGluZy5haSIsInVzZXJSb2xlIjoiYWRtaW4iLCJvcmdJZCI6MSwicmFuayI6bnVsbCwic3NvVXNlciI6ZmFsc2UsInN0cmlwZVVzZXJJZCI6bnVsbCwiZXhwaXJhdGlvbkRhdGUiOm51bGwsInJlYWRPbmx5IjpmYWxzZSwiaW50ZXJuYWwiOmZhbHNlLCJhY2NvdW50X2lkIjoiYjJkNzk5MjAtOWJlMy0xZTAwLTAwY2UtNTIzY2ZjODM5MDEwIiwiYnVja2V0IjoibGFuZGluZ2xlbnMtY3VzdG9tZXItZGF0YS1wcm9kdWN0aW9uIn0sIl9leHBpcmUiOjE2OTc4MTQyMzk1MTIsIl9tYXhBZ2UiOjI1OTIwMDAwMDB9; koa:sess.sig=6usPw7pexyxqpDikODfh0WfSxvA; _gid=GA1.2.1599272570.1695524301; _ga=GA1.2.1062890348.1676512102; _ga_5CP4X6Q718=GS1.1.1695536454.162.1.1695536472.0.0.0',
    }
    b64_pattern = base64.b64encode(list_pattern.encode('utf-8')).decode('utf-8')
    target_dir = f'{tmp_dir}-{b64_pattern}'

    async def upload_pictures(session, data, retry=5):
        with open(f'{target_dir}/{data["file_name"]}', 'rb') as f:
            form_data = {
                "project_id": data['project_id'],
                "dataset_id": data['dataset_id'],
                "file": f,
            }
            async with session.post(landing_upload_api_path, data=form_data, headers=headers) as resp:
                text = await resp.text()
                print('xxxxx status', resp.status, 'text', text, 'xxxxx')
                if resp.status != 200 and resp.status != 201:
                    if retry > 0:
                        await upload_pictures(session, data, retry - 1)
                return resp.status, text, data['file_name']

    async with aiohttp.ClientSession() as session:
        tasks = []
        for file in os.listdir(f'{target_dir}'):
            d = {
                "project_id": '45646308812815',
                "dataset_id": '46767',
                "file_name": file,
            }
            tasks.append(asyncio.ensure_future(upload_pictures(session, d)))
        res = await asyncio.gather(*tasks)
        errs = {}
        for status, text, filename in res:
            print('xxxxx status', status, 'text', text, 'xxxxx')
            if status != 200 and status != 201:
                errs[filename] = text
        if len(errs) > 0:
            st.write('## Upload Error')
            st.error(errs)

    print('xxxxx done upload_to_landing_async to landing xxxxx')



'''
## List a Snowflake Stage
'''
@st.cache_data
def run_list(list_pattern):
    list_sql = f"""
    LIST @tutorial_db.data_schema.zhouyang_test_stage2
    PATTERN = '{list_pattern}';
    """
    res = run_query(list_sql)
    df = pd.DataFrame(res, columns=['name', 'size_bytes', 'MD5', 'last_modified'])
    return df

list_pattern = st.text_input('List Pattern (regex)', '''.*''')



df = run_list(list_pattern)
df


@st.cache_data
def run_get(list_pattern):
    b64_pattern = base64.b64encode(list_pattern.encode('utf-8')).decode('utf-8')
    target_dir = f'{tmp_dir}-{b64_pattern}'
    if not os.path.exists(f'{target_dir}'):
        os.mkdir(f'{target_dir}', 0o777)
    get_sql = f"""
    GET @tutorial_db.data_schema.zhouyang_test_stage2 file://{target_dir}
        PARALLEL = 10
        PATTERN = '.*/{list_pattern}';
    """
    res = run_query(get_sql)
    return res

run_get(list_pattern)

images = []
captions = []
b64_pattern = base64.b64encode(list_pattern.encode('utf-8')).decode('utf-8')
target_dir = f'{tmp_dir}-{b64_pattern}'
for file in os.listdir(f'{target_dir}'):
    image = Image.open(os.path.join(f'{target_dir}', file))
    images.append(image)
    captions.append(file)
st.image(images, width=200, caption=captions)


bt = st.button('Upload Files To Landing', on_click=upload_to_landing, args=(list_pattern,))