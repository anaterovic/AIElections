import os
from enum import Enum

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RUNAAI_SUBSCRIPTION_KEY = os.environ.get("RUNAAI_SUBSCRIPTION_KEY")

class GptModelEnum(str, Enum):
    GPT_35_TURBO_16K = "gpt-3.5-turbo"
    GPT_4_8K = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    YUGO_GPT = "yugogpt"

class RunaAI(OpenAI):
    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Subscription-Key": f"{self.api_key}"}

run_ai_client = RunaAI(
    api_key=RUNAAI_SUBSCRIPTION_KEY,
    base_url='https://api.runaai.com/v1/',
)

def get_response(prompt_token_size, temperature, max_response_size, messages, timeout=60):
    """Util method used to fetch response from ChatGPT API for the given input"""
    openai.api_key = os.environ["OPENAPI_SECURITY_TOKEN"]
    model = os.environ["CHATGPT_MODEL"]

    if model == "GPT-3.5":
        if prompt_token_size < 16000:
            selected_model = GptModelEnum.GPT_35_TURBO_16K
        else:
            raise NotImplementedError("Error - context windows longer than 16K tokens are not supported by GPT-3.5 model")
    elif model == "GPT-4":
        if prompt_token_size > 8000:
            raise NotImplementedError("Error - input texts with more than 8000 tokens are not supported by currently available models")
        else:
            selected_model = GptModelEnum.GPT_4_8K
    elif model == "YUGO_GPT":
        selected_model = GptModelEnum.YUGO_GPT
    else:
        raise NotImplementedError(f"Error! GPT model {model} not supported!")

    print(f"timeout: {timeout}")
    if model == "YUGO_GPT":
        client = run_ai_client
    else:
        client = openai

    response = client.chat.completions.create(
        model=selected_model, messages=messages, timeout=timeout, temperature=temperature
    )

    # print(response)
    return response.choices[0].message.content.strip()

def count_tokens(prompt: str) -> int:
    """Count the number of tokens in the given input prompt. Count is based on token encoding for gpt-3.5-turbo model"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(prompt))

def prompt(system_prompt, user_prompt, temperature=0, max_response_size=200, timeout=600):
    """Returns a response form ChatGPT based on the given prompt."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_prompt = system_prompt + user_prompt
    prompt_token_size = count_tokens(full_prompt) + max_response_size
    return get_response(prompt_token_size, temperature, max_response_size, messages, timeout=timeout)
