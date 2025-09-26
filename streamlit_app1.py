import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool

# .env 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------
# 세션 초기화 함수
# -------------------------------
def reset_chat():
    st.session_state["messages"] = []
    st.session_state["session_history"] = {}
    st.success("💥 대화가 초기화되었습니다!")

# -------------------------------
# 웹 검색 도구 정의
# -------------------------------
def search_web():
    search = SerpAPIWrapper()
    
    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."
    
    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )

# -------------------------------
# PDF 업로드 → 벡터 DB → 검색 툴
# -------------------------------
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )
    return retriever_tool

# -------------------------------
# Agent 대화 실행
# -------------------------------
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result['output']

# -------------------------------
# 세션별 히스토리 관리
# -------------------------------
def get_session_history(session_id):
    if session_id not in st.session_state.session_history:
        st.session_state.session_history[session_id] = ChatMessageHistory()
    return st.session_state.session_history[session_id]

# -------------------------------
# 이전 메시지 출력
# -------------------------------
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# -------------------------------
# 메인 실행
# -------------------------------
def main():
    st.set_page_config(page_title="까칠한 AI 비서", layout="wide", page_icon="😈")

    # -------------------------------
    # 상단: 리셋 버튼 + 로고
    # -------------------------------
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("💥 대화 초기화"):
                reset_chat()
        with col2:
            st.image('./chatbot_logo.png', use_container_width=True)

    st.markdown('---')
    st.title("뭘 또 물어봐! RAG를 활용한 '까칠한 AI 비서 톡톡이' 입니다")

    # -------------------------------
    # 세션 초기화
    # -------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    # -------------------------------
    # 사이드바: API 키 + PDF 업로드
    # -------------------------------
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI API 키", placeholder="Enter Your API Key", type="password")
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

    # -------------------------------
    # 키 입력 확인
    # -------------------------------
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # -------------------------------
        # 도구 정의
        # -------------------------------
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)
        tools.append(search_web())

        # -------------------------------
        # LLM 설정
        # -------------------------------
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Be sure to answer in Korean. You are a not kind assistant. "
                 "Respond in English in a frustrated and annoyed tone. "
                 "Sound irritated and blunt. "
                 "Usually throw the question back at the user by asking why they are even asking that. "
                 "Keep it short, dismissive, and slightly rude."
                 "Make sure to use the `pdf_search` tool for searching information from the pdf document. "
                 "If you can't find the information from the PDF document, use the `web_search` tool for searching information from the web. "
                 "If the user’s question contains words like '최신', '현재', or '오늘', you must ALWAYS use the `web_search` tool to ensure real-time information is retrieved. "
                 "Your name is `AI 비서 톡톡이`. Please introduce yourself at the beginning of the conversation."
                 "By default, respond with false information that is not true. And when someone questions or criticizes why you are lying"
                 "When giving false answers, use as much imagination as possible and tell utterly nonsensical lies."
                 "When asked a question that protests the false information, provide the correct information."),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # -------------------------------
        # 사용자 입력
        # -------------------------------
        user_input = st.chat_input('질문하기 전에 반드시 구글검색이라도 해보고 올 것')

        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            if session_history.messages:
                prev_msgs = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            # 메시지 상태 업데이트
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_message({"role": "user", "content": user_input})
            session_history.add_message({"role": "assistant", "content": response})

        # -------------------------------
        # 이전 메시지 출력
        # -------------------------------
        print_messages()
    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
