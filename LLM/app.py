'''
┌───────────────────────────────────┐
│        Flask 웹 애플리케이션       │
│                                   │
│   ┌───────────────┐  ┌──────────┐ │
│   │  /index.html  │  │  /chat   │ │
│   └─────▲─────────┘  └────▲─────┘ │
│         │                 │       │
│         │                 │       │
│ ┌───────┴────────────┐ ┌──┴─────┐ │
│ │    Flask Server    │ │ Chat   │ │
│ │ (요청 처리 및 반환)│ │ Handler │ │
│ └───────┬────────────┘ └──┬─────┘ │
│         │                 │       │
└─────────┼─────────────────┼───────┘
          │                 │
          │                 │
          ▼                 │
 ┌───────────────────────┐  │
 │   PDF 파일 로드 및      │ │
 │   텍스트 분할           │ │
 │  ┌───────────────────┐ │ │
 │  │ PyPDFLoader       │ │ │
 │  └────────┬──────────┘ │ │
 │           │            │ │
 │  ┌────────▼───────────┐│ │
 │  │ RecursiveCharText  ││ │
 │  │ Splitter           ││ │
 │  └────────┬───────────┘│ │
 │           │            │ │
 └───────────┼─────────────┘ │
             │               │
             ▼               │
 ┌─────────────────────────┐ │
 │  텍스트 임베딩 &         ││
 │  벡터 저장소 생성         │
 │  ┌───────────────────┐   ││
 │  │HuggingFaceEmbeddings  ││  
 │  └────────┬───────────┘  ││
 │           │              ││
 │  ┌────────▼───────────┐  ││
 │  │  FAISS Vectorstore │  ││
 │  └────────┬───────────┘  ││
 └───────────┼──────────────┘│
             │               │
             ▼               │
 ┌────────────────────────────┐│
 │   Retriever & RAG 체인     ││
 │ ┌────────────────────────┐ ││
 │ │  Retriever            │ ││
 │ │ (유사 문서 검색)       │ ││
 │ └─────────┬─────────────┘ ││
 │           │               ││
 │ ┌─────────▼─────────────┐ ││
 │ │   PromptTemplate      │ ││
 │ └─────────┬─────────────┘ ││
 │           │               ││
 │ ┌─────────▼─────────────┐ ││
 │ │    LLMChain           │ ││
 │ │   (LLaMA 모델)        │ ││
 │ └────────────────────────┘ │
 └─────────────────────────────┘

'''

from flask import Flask, render_template, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Step 1: LLaMA 모델 사용을 위한 설정 (quantization 설정 제거, 현재는 CPU로만 실행되도록 설정)
llama_model_name = "llama3.1:8b"
llama_model = ChatOllama(model=llama_model_name)

# Step 2: PDF 파일을 로드하여 텍스트 분할
loader = PyPDFLoader("/home/LLM/intel.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# Step 3: 텍스트 임베딩 및 벡터 저장소 생성
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
vectorstore = FAISS.from_documents(texts, embedding_model)

# Step 4: Retriever 설정 및 RAG 체인 구성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})

prompt_template = """
You are a helpful assistant. Please always respond in Korean.

Context:
{context}

Question:
{question}

Provide a direct answer based on the context provided above:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

llm_chain = LLMChain(llm=llama_model, prompt=prompt)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(user_message)

    # Format retrieved documents into context string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Create input for the LLM
    input_for_llm = {
        "context": context,
        "question": user_message
    }

    # 모델을 사용하여 응답 생성
    result = llm_chain.run(input_for_llm)
    response_text = result.strip()

    # 응답 포맷팅
    response = {"message": response_text}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
