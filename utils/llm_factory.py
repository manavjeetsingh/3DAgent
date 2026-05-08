from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from utils.key_manager import get_or_prompt_key


def create_llm(llm_option: dict, streaming: bool = True) -> BaseChatModel:
    provider = llm_option["provider"]
    model = llm_option["model"]
    key_name = llm_option["key_name"]
    provider_name = llm_option["name"]

    api_key = get_or_prompt_key(key_name, provider_name)

    kwargs: dict[str, Any] = {"streaming": streaming}

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, **kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, mistral_api_key=api_key, **kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, groq_api_key=api_key, **kwargs)

    if provider == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(model=model, cohere_api_key=api_key)

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            **kwargs,
        )

    if provider == "xai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            **kwargs,
        )

    # Generic OpenAI-compatible provider
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        base_url = llm_option.get("base_url", "")
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
