import subprocess

try:
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    print(f"Playwright browser install failed: {e}")

import os
import asyncio
import json
import streamlit as st
from pydantic import create_model, Field
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dotenv import load_dotenv
import os

load_dotenv(".env")

# 비밀번호 설정 (보안상 .env 에서 불러오는 게 좋음)
PASSWORD = os.getenv("APP_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def password_gate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 Password Required")
        pw = st.text_input("Enter the password", type="password")
        if pw == PASSWORD:
            st.session_state.authenticated = True
            st.success("Access granted!")
            st.rerun()
        elif pw:
            st.error("Incorrect password")
        st.stop()


password_gate()

# 🧠 사용자로부터 API Key 입력받기
if not OPENAI_API_KEY:
    st.warning("⚠️ API key가 입력되지 않았습니다.")


def run_crawler(schema_json: str, url: str, instruction: str, model_choice: str):
    """
    schema_json: Pydantic 모델의 JSON 스키마 문자열
    url: 크롤링할 대상 URL
    instruction: 데이터 추출에 사용할 instruction
    # model_choice: 선택된 LLM 모델 (예: "gpt-4o" 또는 "gpt-4o-mini")
    """

    async def main():
        # LLM Extraction Strategy 설정 (선택된 모델 반영)
        llm_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(provider=f"{model_choice}", api_token=OPENAI_API_KEY),
            schema=schema_json,
            extraction_type="schema",
            instruction=instruction,
            chunk_token_threshold=1000,
            overlap_rate=0.0,
            apply_chunking=True,
            input_format="markdown",
            extra_args={"temperature": 0.0, "max_tokens": 800},
        )

        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            cache_mode=CacheMode.BYPASS,
            page_timeout=120000,
        )

        browser_cfg = BrowserConfig(
            headless=True, verbose=True, user_agent_mode="random"
        )

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success:
                data = json.loads(result.extracted_content)
                return data, llm_strategy
            else:
                return {"error": result.error_message}, None

    return asyncio.run(main())


def main():
    st.title("Dynamic Model & Crawler Example")

    # 메인 화면: 크롤링 URL 입력
    url = st.text_input("Enter URL to crawl", "https://example.com/products")

    # 사이드바: 모델 생성 및 extraction instruction 입력
    st.sidebar.title("Model & Extraction Settings")

    # Extraction Instruction 입력 (사이드바)
    instruction = st.sidebar.text_area(
        "Extraction Instruction",
        "Extract all product objects with 'name' and 'price' from the content.",
    )

    # LLM 모델 선택 (사이드바)
    model_choice = st.sidebar.selectbox(
        "Select LLM Model", ["openai/gpt-4o", "openai/gpt-4o-mini"], index=0
    )

    # 사이드바: 동적 필드 입력
    st.sidebar.header("Dynamic Model Fields")
    if "row_count" not in st.session_state:
        st.session_state.row_count = 1
    if "DynamicModel" not in st.session_state:
        st.session_state.DynamicModel = None

    for i in range(1, st.session_state.row_count + 1):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.text_input(f"Field Name {i}", key=f"field_name_{i}")
        with col2:
            st.text_input(f"Field Type {i}", key=f"field_type_{i}", value="str")

    # "Add Field Pair" 버튼: 필드 쌍 추가
    def add_row():
        st.session_state.row_count += 1

    # "Generate Model" 버튼: 입력된 필드로 Pydantic 모델 생성
    def generate_model():
        dynamic_fields = {}
        for i in range(1, st.session_state.row_count + 1):
            field_name = st.session_state.get(f"field_name_{i}", "").strip()
            field_type_str = st.session_state.get(f"field_type_{i}", "").strip()
            if not field_name:
                continue

            # 간단한 타입 매핑
            if field_type_str.lower() == "int":
                python_type = int
            elif field_type_str.lower() == "float":
                python_type = float
            elif field_type_str.lower() == "bool":
                python_type = bool
            else:
                python_type = str

            dynamic_fields[field_name] = (
                python_type,
                Field(..., description=f"{field_name} (type: {field_type_str})"),
            )
        if dynamic_fields:
            NewModel = create_model("DynamicModel", **dynamic_fields)
            st.session_state.DynamicModel = NewModel
            st.sidebar.success("Pydantic 모델이 생성되었습니다!")
        else:
            st.sidebar.warning("유효한 필드가 없습니다. 필드 이름을 입력하세요.")

    st.sidebar.button("Add Field Pair", on_click=add_row)
    st.sidebar.button("Generate Model", on_click=generate_model)

    if st.session_state.DynamicModel:
        st.sidebar.write("**Generated Model Schema**")
        st.sidebar.json(st.session_state.DynamicModel.model_json_schema())

    # 메인 화면: 크롤러 실행
    if st.button("Run Crawler"):
        if st.session_state.DynamicModel:
            schema_json = st.session_state.DynamicModel.model_json_schema()
            with st.spinner("Crawling... please wait..."):
                data, strategy = run_crawler(
                    json.dumps(schema_json), url, instruction, model_choice
                )
            if "error" in data:
                st.error(f"Crawling error: {data['error']}")
            else:
                st.success("Crawling completed!")
                st.json(data)
                if strategy:
                    st.write("**Token Usage**")
                    strategy.show_usage()
        else:
            st.warning("먼저 모델을 생성하세요.")


if __name__ == "__main__":
    main()
