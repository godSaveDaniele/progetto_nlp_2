import json
from asyncio import sleep

import httpx

from ..logger import logger
from .client import Client

# Preferred provider routing arguments.
# Change depending on what model you'd like to use.
PROVIDER = {"order": ["Together", "DeepInfra"]}


class Response:
    def __init__(self, response):
        self.text = response


class OpenRouter(Client):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url="https://openrouter.ai/api/v1/chat/completions",
    ):
        super().__init__(model)

        self.headers = {"Authorization": f"Bearer {api_key}"}

        self.url = base_url
        timeout_config = httpx.Timeout(5.0)
        self.client = httpx.AsyncClient(timeout=timeout_config)

    def postprocess(self, response):
        response_json = response.json()
        msg = response_json["choices"][0]["message"]["content"]
        return Response(msg)
    
    """

    async def generate(  # type: ignore
        self, prompt: str, raw: bool = False, max_retries: int = 1, **kwargs  # type: ignore
    ) -> Response:  # type: ignore
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", 500)
        temperature = kwargs.pop("temperature", 1.0)
        data = {
            "model": self.model,
            "messages": prompt,
            # "provider": PROVIDER,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers
                )
                if raw:
                    return response.json()
                result = self.postprocess(response)

                return result

            except json.JSONDecodeError:
                logger.warning(
                    f"Attempt {attempt + 1}: Invalid JSON response, retrying..."
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {repr(e)}, retrying...")

            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")


        """
    
    async def generate(self, prompt: str, raw: bool = False, max_retries: int = 3, **kwargs) -> Response:
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 1.0)

        data = {
            "model": self.model,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                logger.info(f"[Attempt {attempt + 1}] Sending request to OpenRouter...")
                response = await self.client.post(url=self.url, json=data, headers=self.headers)

                logger.info(f"[Attempt {attempt + 1}] Status code: {response.status_code}")
                logger.debug(f"[Attempt {attempt + 1}] Raw response content: {response.text}")

                if response.status_code != 200:
                    logger.warning(f"[Attempt {attempt + 1}] Non-200 response: {response.status_code}")
                    continue

                if raw:
                    return response.json()

                try:
                    result = self.postprocess(response)
                    logger.debug(f"[Attempt {attempt + 1}] Postprocessed text: {result.text}")

                    if not result.text or not result.text.strip():
                        logger.warning(f"[Attempt {attempt + 1}] Empty response text, retrying...")
                        continue

                    return result

                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    logger.warning(f"[Attempt {attempt + 1}] JSON structure error: {e}")

            except Exception as e:
                logger.warning(f"[Attempt {attempt + 1}] Exception during request: {repr(e)}")

            await sleep(1)

        logger.error("All retry attempts failed. Final error raised.")
        raise RuntimeError("Failed to generate text after multiple attempts.")

