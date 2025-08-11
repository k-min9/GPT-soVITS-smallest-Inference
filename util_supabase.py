'''
pip install supabase

사용 프로젝트 : jsons2
사용 Storgage : json_bucket
'''
import os
import json
from supabase import create_client, Client

from kei import SUPABASE_URL, SUPABASE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#  URL을 받아 JSON 파일로 저장한 후 Supabase 스토리지에 업로드
# def post_ngrok_path(url, status="open", id='temp'):
#     file_name = "my_little_jarvis_plus_ngrok_server.json"
#     bucket_name = "json_bucket"

#     # JSON 데이터를 파일로 저장
#     json_data = {
#         "url": url,
#         "status": status
#         }
#     os.makedirs('./log', exist_ok=True)
#     local_file_path = f"./log/{file_name}"

#     with open(local_file_path, "w", encoding="utf-8") as f:
#         json.dump(json_data, f, ensure_ascii=False, indent=4)

#     # Supabase 스토리지에 파일 업로드
#     with open(local_file_path, "rb") as f:
#         response = supabase.storage.from_(bucket_name).upload(
#             file=f,
#             path=file_name,
#             file_options={"cache-control": "no-cache", "upsert": "true"},  # 파일 덮어쓰기 설정
#         )

#     if isinstance(response, dict) and response.get("error"):
#         print("Error uploading file:", response["error"])
#     else:
#         print(f"File {file_name} successfully uploaded or updated.")


def post_clear_ngrok_path():
    file_name = "my_little_jarvis_plus_ngrok_server.json"
    bucket_name = "json_bucket"
    
    data = {}
    
    # JSON 데이터를 파일로 저장
    os.makedirs('./log', exist_ok=True)
    local_file_path = f"./log/{file_name}"

    with open(local_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Supabase 스토리지에 파일 업로드
    try:
        with open(local_file_path, "rb") as f:
            response = supabase.storage.from_(bucket_name).upload(
                file=f,
                path=file_name,
                file_options={"cache-control": "no-cache", "upsert": "true"},  # 파일 덮어쓰기 설정
            )

        if isinstance(response, dict) and response.get("error"):
            print("Error uploading file:", response["error"])
        else:
            print(f"File {file_name} successfully cleared or uploaded.")

    except Exception as e:
        print(f"Error while uploading file: {e}")

# URL을 받아 JSON 파일로 저장한 후 Supabase 스토리지에 업로드
def post_ngrok_path(url, status="open", id='temp'):
    file_name = "my_little_jarvis_plus_ngrok_server.json"
    bucket_name = "json_bucket"

    # 기존 데이터를 가져오기
    existing_data = get_ngrok_path()

    existing_data[id] = {
        "url": url,
        "status": status
    }

    # JSON 데이터를 파일로 저장
    os.makedirs('./log', exist_ok=True)
    local_file_path = f"./log/{file_name}"

    with open(local_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # Supabase 스토리지에 파일 업로드
    try:
        with open(local_file_path, "rb") as f:
            response = supabase.storage.from_(bucket_name).upload(
                file=f,
                path=file_name,
                file_options={"cache-control": "no-cache", "upsert": "true"},  # 파일 덮어쓰기 설정
            )

        if isinstance(response, dict) and response.get("error"):
            print("Error uploading file:", response["error"])
        else:
            print(f"File {file_name} successfully uploaded or updated.")

    except Exception as e:
        print(f"Error while uploading file: {e}")

# 테스트용 : Supabase에서 JSON 파일을 다운로드하고 내용을 출력
def get_ngrok_path():
    file_name = "my_little_jarvis_plus_ngrok_server.json"
    bucket_name = "json_bucket"
    os.makedirs('./log', exist_ok=True)
    local_file_path = f"./log/{file_name}"

    # Supabase 스토리지에서 파일 다운로드
    try:
        response = supabase.storage.from_(bucket_name).download(file_name)
        if response and isinstance(response, bytes):
            data = json.loads(response.decode("utf-8"))
            return data
        else:
            print("Failed to retrieve data from bucket.")
            return {}
    except Exception as e:
        print(f"Error while fetching data: {e}")
        return {}

    # # Test용 로직
    # # 다운로드된 내용을 로컬 파일에 저장
    # with open(local_file_path, "wb") as f:
    #     f.write(response)

    # # 저장된 파일을 읽어 JSON 데이터 파싱 및 출력
    # with open(local_file_path, "r", encoding="utf-8") as f:
    #     json_data = json.load(f)
    #     print(json_data)

# 544 에러시 기존 버켓 삭제 후 재생성
def recreate_bucket():
    bucket_name = "json_bucket"    
    try:
        # 1. 기존 버킷 삭제
        delete_response = supabase.storage.delete_bucket(bucket_name)
        if isinstance(delete_response, dict) and delete_response.get("error"):
            print(f"Error deleting bucket {bucket_name}: {delete_response['error']}")
        else:
            print(f"Bucket {bucket_name} successfully deleted.")

        # # 2. 새로운 버킷 생성
        # create_response = supabase.storage.create_bucket(bucket_name)
        # if isinstance(create_response, dict) and create_response.get("error"):
        #     print(f"Error creating bucket {bucket_name}: {create_response['error']}")
        # else:
        #     print(f"Bucket {bucket_name} successfully created.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # recreate_bucket()
    
    # 테스트용 업로드 및 다운로드 호출
    post_clear_ngrok_path()
    post_ngrok_path("https://test-url2.com")
    print(get_ngrok_path())
    post_ngrok_path("https://test-urlX.com", id='test')
    print(get_ngrok_path())
