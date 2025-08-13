"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

import os
import httpx
import openai
import chromadb
import weaviate
from pydantic import BaseModel
from typing import Optional, Any, Dict
from llama_index.llms.openai_like import OpenAILike
from typing import List, Union, Generator, Iterator
from weaviate.config import AdditionalConfig, Timeout
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference


class CustomTextEmbeddingsInference(TextEmbeddingsInference):
    client: openai.OpenAI | None = None
    async_client: openai.AsyncOpenAI | None = None
    batch_size: int = 32

    def __init__(self, batch_size: int, api_key: str, default_headers: Dict[str, str], **kwargs):
        super().__init__(**kwargs)

        self.auth_token = api_key.strip()
        if self.auth_token.lower().startswith("bearer "):
            self.auth_token = self.auth_token[7:].strip()
        self.batch_size = batch_size
        self.timeout = 300
        #print(f"Initializing CustomTextEmbeddingsInference with base_url: {self.base_url}, auth_token: {self.auth_token}", flush=True)  
        try:
            self.client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.auth_token,
                http_client=httpx.Client(verify=False),
                default_headers=default_headers,
            )
            self.async_client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.auth_token,
                http_client=httpx.AsyncClient(verify=False),
                default_headers=default_headers,
            )
        except Exception as e:
            # logging.error(f"Failed to initialize OpenAI client(s): {e}")
            raise str(e)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(6):
            try:
                result = self.client.embeddings.create(
                    model=self.model_name,
                    extra_body={"input_type": "query"},
                    input=texts,
                )
                if hasattr(result, "response") and getattr(result.response, "status_code", 200) != 200:
                    raise Exception(f"Status code: {result.response.status_code}")
                return [item.embedding for item in result.data]
            except Exception as e:
                # logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
                time.sleep(5)
        raise Exception("Failed to get embeddings after 6 attempts")

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        for attempt in range(6):
            try:
                result = await self.async_client.embeddings.create(
                    model=self.model_name,
                    extra_body={"input_type": "query"},
                    input=texts,
                )
                if hasattr(result, "response") and getattr(result.response, "status_code", 200) != 200:
                    raise Exception(f"Status code: {result.response.status_code}")
                return [item.embedding for item in result.data]
            except Exception as e:
                # logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
                await asyncio.sleep(5)
        raise Exception("Failed to get embeddings after 6 attempts")


def get_openai_embedding(
    embedding_url: str, embedding_key: str, plcfg: Dict[str, Any]= {}
) -> OpenAIEmbedding:
    return OpenAIEmbedding(model=embedding_url, api_key=embedding_key)


def get_embed_model(emburl: str, embkey: str) -> CustomTextEmbeddingsInference:
    emburl = self.valves.emb_end_point
    embkey = self.valves.emb_api_key

    if not emburl.startswith("http"):
        try:
            return get_openai_embedding(emburl, embkey, plcfg)
        except:
            try:
                return HuggingFaceEmbedding(model_name=emburl)
            except Exception as e:
                error_msg = f"Error while fetching embedding model. {e}"
                raise Exception(error_msg)

    try:
        emb_inference = CustomTextEmbeddingsInference(
            model_name=emb_model,
            base_url=base_url,
            text_instruction=" ",
            query_instruction=" ",
            truncate_text=False,
            batch_size=batch_size,
            api_key=api_key,
            default_headers=headers
        )
    except Exception as e:
        error_msg = f"Error while fetching embedding model. {e}"
        raise Exception(error_msg)

    return emb_inference



class Pipeline:
    class Valves(BaseModel):
        llm_end_point: Optional[str] = "dummy"
        llm_api_key: Optional[str] = "dummy"
        emb_end_point: Optional[str] = "dummy"
        emb_api_key: Optional[str] = "dummy"
        # openai_apikey: Optional[str] = "dummy-openai-key"
        user_prompt: Optional[str] = "either-path-or-string"
        dataset: str = "dummy"
        textkey: str = "paperdocs"
        top_k: int = 3


    def __init__(self):
        self.index: Any = None
        self.retriever: Any = None
        self.user_prompt: str = None
        self.nodes: List[Any] = None
        self.documents: List[Any] = None
        self.chat_engine: SimpleChatEngine = None
        self.valves: Dict[str, Any] = self.Valves(
            **{
                "llm_end_point": os.getenv("LLM_END_POINT", "dummy-llm-end-point"),
                "llm_api_key": os.getenv("LLM_API_KEY", "dummy-llm-api-key"),
                # "openai_apikey": os.getenv("APIKEY", "dummy-openai-key"),
                "user_prompt": os.getenv("USER_PROMPT", "either-path-or-string"),
            }
        )

    def get_model_name(self,):
        headers = {"Authorization": self.valves.llm_api_key}

        with httpx.Client(verify=False) as _htx_cli:
            client = openai.OpenAI(
                base_url=self.valves.llm_end_point,
                api_key=self.valves.llm_api_key,
                default_headers=headers,
                http_client=_htx_cli,
            )

            models = client.models.list()

            # Case 1: Check if model_id is directly accessible
            if hasattr(models, "model_id"):
                return models.model_id

            # Case 2: Check if it's in the dict format
            elif hasattr(models, "dict"):
                model_dict = models.dict()
                if "data" in model_dict and len(model_dict["data"]) > 0:
                    return model_dict["data"][0]["id"]

            # If neither case works, raise an exception
            raise ValueError(
                "Unable to retrieve model ID. API response format may have changed."
            )

    def get_chat_engine(self, ):
        llmbase = self.valves.llm_end_point
        headers = {"Authorization": self.valves.llm_api_key}

        _htx_cli = httpx.Client(verify=False) if llmbase.startswith("https") else None
        _htx_acli = (
            httpx.AsyncClient(verify=False) if llmbase.startswith("https") else None
        )

        llm = OpenAILike(
            model=self.get_model_name(),
            api_base=llmbase,
            api_key=self.valves.llm_api_key,
            temperature=0,
            is_chat_model=True,
            # max_tokens=max_tokens,
            max_tokens=2048,
            default_headers=headers,
            http_client=_htx_cli,
            async_http_client=_htx_acli,
        )

        self.chat_engine = SimpleChatEngine.from_defaults(llm=llm)


    def get_prompt(self):
        try:
            with open(self.valves.user_prompt, "r") as f:
                self.user_prompt = f.read()
        except:
            self.user_prompt = self.valves.user_prompt

    def set_prompt(self,query: str) -> str:
        text_list = [ node.text for node in self.nodes ]
        context = "\n".join(text_list)
        prompt = self.user_prompt.format(context=context, question=query)
        return prompt
       
    def get_weaviate_client(self):
        client = None
        try:
            http_host = os.getenv("WEAVIATE_SERVICE_HOST", "weaviate.d3x.svc.cluster.local")

            http_grpc_host = os.getenv(
                "WEAVIATE_SERVICE_HOST_GRPC", "weaviate-grpc.d3x.svc.cluster.local"
            )

            http_port = os.getenv("WEAVIATE_SERVICE_PORT", "80")
            grpc_port = os.getenv("WEAVIATE_GRPC_SERVICE_PORT", "50051")

            api_key = os.getenv("DKUBEX_API_KEY", "")

            if os.getenv("WEAVIATE_AUTH_APIKEY", None) != None:
                api_key = os.environ["WEAVIATE_AUTH_APIKEY"]

            additional_headers = {"authorization": f"Bearer {api_key}"}
            if "://" in http_host:
                http_host = http_host.split("://")[1]
            client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=int(http_port),
                http_secure=False if int(http_port) == 80 else True,
                grpc_host=http_grpc_host,
                grpc_port=int(grpc_port),
                grpc_secure=False if int(grpc_port) != 443 else True,
                headers=additional_headers,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=120, query=300, insert=300)  # Values in seconds
                ),
            )

            if not client.is_ready():
                raise ConnectionError("Weaviate instance not ready")

            return client

        except Exception as e:
            # logging.error(f"Error in creating Weaviate Client. Error: {str(e)}")
            if client:
                client.close()
            raise Exception(f"Error in creating Weaviate Client. Error: {str(e)}")

    async def on_startup(self):

        # Set the OpenAI API key
        # os.environ["OPENAI_API_KEY"] = os.getenv("APIKEY")
        # os.environ["OPENAI_API_KEY"] = self.valves.openai_apikey
        # weaviate_uri: Optional[str] = os.getenv("WEAVIATE_URI", "dummy-uri")
        # weaviate_auth_apikey: Optional[str] = os.getenv("WEAVIATE_AUTH_APIKEY", "dummy-auth-key"),

        # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_APIKEY")
        # self.documents = SimpleDirectoryReader("/home/kiran-vijapure/pipelines/maha").load_data()

        # self.index = VectorStoreIndex.from_documents(self.documents)
        # self.retriever = self.index.as_retriever(similarity_top_k=3)

        # Get weaviate vector store index.
        self.weaviate_client = self.get_weaviate_client()
        # vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=class_name, text_key=textkey)


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    def get_weaviate_retriever(self):
        embed_model = get_embed_model(self.valves.emb_end_point, self.valves.emb_api_key)

        vector_store = WeaviateVectorStore(
                        weaviate_client=weaviate_client, 
                        index_name=f"D{self.valves.dataset}chunks",
                        text_key=self.valves.textkey
                    )
        vector_index = VectorStoreIndex.from_vector_store(
                        vector_store, 
                        show_progress=True, 
                        embed_model=embed_model
                    )
        self.retriever = vector_index.as_retriever(similarity_top_k=self.valves.top_k)


    def get_nodes(self, user_message):
        if not user_message.startswith("### Task"):
            self.nodes = self.retriever.retrieve(user_message)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # print(messages)
        # print(user_message)

        self.get_weaviate_retriever()
        self.get_chat_engine()
        self.get_prompt()

        self.get_nodes(user_message)
        prompt = self.set_prompt(user_message)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        response = self.chat_engine.stream_chat(prompt)

        return response.response_gen


