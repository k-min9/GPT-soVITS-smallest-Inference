'''
set Tunnel of server
'''
from pyngrok import ngrok
import util_supabase

from kei import PYNGROK_KEY

def start_ngrok(id='temp', key=""):
    if not key:
        key = PYNGROK_KEY
    # print('id :', id, '/ ngrok key :', key)
    try:
        # ngrok 터널 생성
        ngrok.set_auth_token(key) 
        http_tunnel = ngrok.connect(5010)  # Flask 서버의 포트를 연결
        print(f"ngrok public URL: {http_tunnel.public_url}")
        util_supabase.post_ngrok_path(http_tunnel.public_url, status="open", id=id)
        # return http_tunnel.public_url  # 어디서 쓸것 같지는 않은데...
    except:
        print('Fail to make ngrok tunnel')
        print('Making Local Server at port 5000...')


if __name__ == "__main__":
    # pyngrok 터널링과 Flask 서버 실행을 분리
    start_ngrok()
