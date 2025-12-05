import json
import logging
import os

import openai

logger = logging.getLogger(__name__)


MODEL_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1": {"input": 3.00, "output": 12.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 150.00, "output": 600.00},
}


def calc_query_cost(usage: dict, model: str) -> float:
    if model not in MODEL_PRICING:
        raise ValueError(f"Model {model} not in MODEL_PRICING table")

    price = MODEL_PRICING[model]
    input_cost = usage["prompt_tokens"] * (price["input"] / 1_000_000)
    output_cost = usage["completion_tokens"] * (price["output"] / 1_000_000)

    return round(input_cost + output_cost, 6)


def query_llm_sync(model, messages, temperature=0.1, max_tokens=1000, **kwargs):
    apikey = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=apikey)

    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs
    )

    usage = response.usage.model_dump()
    logger.info(f"Usage: {usage}")
    cost = calc_query_cost(usage, model)
    logger.info(f"Query cost: ${cost}")

    if "logprobs" in kwargs:
        return response, json.loads(response.choices[0].message.content)

    return json.loads(response.choices[0].message.content)


# Async example
async def query_llm_async(model, messages, temperature=0.1, max_tokens=1000, **kwargs):
    apikey = os.getenv("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(api_key=apikey)

    response = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs
    )
    usage = response.usage.model_dump()
    logger.info(f"Usage: {usage}")
    cost = calc_query_cost(usage, model)
    logger.info(f"Query cost: ${cost}")

    if "logprobs" in kwargs:
        return response, json.loads(response.choices[0].message.content)

    return json.loads(response.choices[0].message.content)
