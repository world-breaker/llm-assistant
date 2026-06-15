"""
rag_langchain.py —— LangChain 版 RAG（Step 2）
==============================================
演示 LangChain 的核心组件，面试时可以对比手写版和框架版。
"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def build_langchain_kb():
    """
    用 LangChain 构建知识库——三行代码替代手写版 100 行
    面试讲: LangChain 的价值——标准化组件，团队协作不需要重新发明轮子
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_dir = os.path.join(project_root, "data", "knowledge")
    db_dir = os.path.join(project_root, "chroma_db_langchain")

    # 1. 加载文档（LangChain 自动处理多种格式）
    print("[LangChain] 加载文档...")
    loader = DirectoryLoader(knowledge_dir, glob="*.md", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    print(f"  加载 {len(docs)} 个文档")

    # 2. 分块（RecursiveCharacterTextSplitter 按段落智能切分）
    print("[LangChain] 分块...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n", "。", "，", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  切分为 {len(chunks)} 块")

    # 3. 向量化 + 存入 Chroma（一行搞定）
    print("[LangChain] 向量化...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=db_dir,
        collection_name="llm_knowledge_lc"
    )
    print(f"  ✓ LangChain 知识库构建完成")
    return db_dir


def demo_langchain_qa():
    """演示 LangChain 的 RetrievalQA 链"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(project_root, "chroma_db_langchain")

    # 加载嵌入模型和向量库
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name="llm_knowledge_lc"
    )

    # 加载 Qwen
    model_path = os.path.join(
        os.environ["USERPROFILE"],
        ".cache", "modelscope", "hub", "models", "Qwen", "Qwen2___5-3B-Instruct"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).cuda()

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=150, do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # RetrievalQA 链：检索 → 拼 prompt → 生成，一行代码
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 把检索到的文档全拼进 prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    # 测试
    questions = [
        "LoRA的原理是什么？",
        "RAG的三个阶段是什么？",
    ]
    for q in questions:
        print(f"\n问题: {q}")
        result = qa_chain.invoke({"query": q})
        print(f"回答: {result['result'][:200]}")
        if result.get("source_documents"):
            print(f"参考: {result['source_documents'][0].metadata.get('source', '?')}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_langchain_kb()
    else:
        # 先建库再演示
        try:
            demo_langchain_qa()
        except Exception:
            print("请先运行: python rag_langchain.py build")
