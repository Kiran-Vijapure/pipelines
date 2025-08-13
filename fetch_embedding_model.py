import httpx
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
            logging.error(f"Failed to initialize OpenAI client(s): {e}")
            raise

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
                logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
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
                logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
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

