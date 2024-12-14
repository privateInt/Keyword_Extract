import streamlit as st
import requests
import json

# Setup session states
def reset_messages():
    st.session_state["messages_op"] = [
        {"role": "assistant", "content": "- 문장 입력시 keyword를 추출해드립니다."}
    ]
    
if "messages_op" not in st.session_state:
    reset_messages()     

def main():
    st.title("KoBART Keyword Extract 요약 Test")
    
    for message in st.session_state["messages_op"]:
        if "hide" not in message:
            with st.chat_message(message["role"]):
                st.markdown(message['content'])
                
    if prompt := st.chat_input("문장을 여기 입력해주세요."):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(text="생각 중입니다.."):
                diction = {
                    'prompt': prompt
                }
                try:
                    response = requests.post("http://localhost:8459/keyword_extract", data=json.dumps(diction))
                    if response:
                        result = json.loads(response.text)
                    else:
                        result = {}
                        result['prompt'] = ''
                        result['result'] = ''
                    st.write(result['result'])
                except requests.exceptions.RequestException as e:
                    st.warning(f"현재 문제가 발생했습니다. 잠시후 다시 시도해 주세요. 문제가 계속되는 경우 관리자에게 연락바랍니다.")
            
        st.session_state["messages_op"].append({"role": "user", "content": prompt})
        st.session_state["messages_op"].append({"role": "assistant", "content": result})
        
    st.button("대화 초기화", on_click=reset_messages)

if __name__=="__main__":
    main()
