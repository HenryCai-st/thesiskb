"""The only place that talks to models. Pipeline code imports from here, never vendor SDKs."""

from thesiskb.llm.gateway import LLMGateway, RerankResult, get_gateway

__all__ = ["LLMGateway", "RerankResult", "get_gateway"]
