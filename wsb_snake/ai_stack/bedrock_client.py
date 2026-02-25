"""
AWS Bedrock Unified Client

Provides access to all Bedrock models used by the Predator Stack:
- Claude Sonnet 4.5 (adversarial + synthesis)
- Nova Micro (speed filter)
- Titan Embeddings V2 (semantic matching)
"""

import json
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import boto3 (optional - may not be installed)
try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    logger.warning("BEDROCK_CLIENT: boto3 not installed - Bedrock features disabled")


@dataclass
class BedrockResponse:
    """Response from a Bedrock model invocation."""
    text: str
    model_id: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


class BedrockClient:
    """
    Unified client for AWS Bedrock model invocations.

    Handles:
    - Model-specific request/response formatting
    - Retries and error handling
    - Latency tracking
    - Cost estimation
    """

    # Model IDs
    MODELS = {
        'claude-sonnet': 'us.anthropic.claude-sonnet-4-5-20250514-v1:0',
        'claude-haiku': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'nova-micro': 'us.amazon.nova-micro-v1:0',
        'nova-pro': 'us.amazon.nova-pro-v1:0',
        'titan-embed': 'amazon.titan-embed-text-v2:0'
    }

    # Cost per 1M tokens (input/output)
    COSTS = {
        'claude-sonnet': (3.00, 15.00),
        'claude-haiku': (0.80, 4.00),
        'nova-micro': (0.000035, 0.00014),
        'nova-pro': (0.0008, 0.0035),
        'titan-embed': (0.02, 0.0)  # Embeddings have no output cost
    }

    def __init__(self, region: str = None):
        """
        Initialize Bedrock client.

        Args:
            region: AWS region (default: us-east-1)
        """
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.client = None
        self._available = False

        # Track usage
        self._call_count = 0
        self._total_latency_ms = 0
        self._total_cost = 0.0

        if not BOTO3_AVAILABLE:
            logger.warning("BEDROCK_CLIENT: boto3 not available, running in mock mode")
            return

        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                config=Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    read_timeout=30,
                    connect_timeout=5
                )
            )
            self._available = True
            logger.info(f"BEDROCK_CLIENT: Initialized in {self.region}")
        except Exception as e:
            logger.warning(f"BEDROCK_CLIENT: Failed to initialize: {e}")

    def is_available(self) -> bool:
        """Check if Bedrock client is available."""
        return self._available and self.client is not None

    def invoke_claude(
        self,
        prompt: str,
        model: str = 'claude-sonnet',
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str = None
    ) -> BedrockResponse:
        """
        Invoke Claude model (Sonnet or Haiku).

        Args:
            prompt: User prompt
            model: 'claude-sonnet' or 'claude-haiku'
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 = deterministic)
            system: Optional system prompt
        """
        if not self.is_available():
            return BedrockResponse(
                text='',
                model_id='',
                latency_ms=0,
                success=False,
                error='Bedrock client not available'
            )

        model_id = self.MODELS.get(model, self.MODELS['claude-sonnet'])

        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system:
            body["system"] = system

        return self._invoke(model_id, body, model)

    def invoke_nova(
        self,
        prompt: str,
        model: str = 'nova-micro',
        max_tokens: int = 100,
        temperature: float = 0.0
    ) -> BedrockResponse:
        """
        Invoke Amazon Nova model (Micro or Pro).

        Args:
            prompt: Input text
            model: 'nova-micro' or 'nova-pro'
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
        """
        if not self.is_available():
            return BedrockResponse(
                text='',
                model_id='',
                latency_ms=0,
                success=False,
                error='Bedrock client not available'
            )

        model_id = self.MODELS.get(model, self.MODELS['nova-micro'])

        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        }

        return self._invoke(model_id, body, model, is_nova=True)

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using Titan Embeddings V2.

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector
        """
        if not self.is_available():
            return []

        model_id = self.MODELS['titan-embed']

        body = {
            "inputText": text,
            "dimensions": 768,
            "normalize": True
        }

        start = time.time()

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            embedding = result.get('embedding', [])

            latency = (time.time() - start) * 1000
            self._track_usage('titan-embed', latency, len(text.split()))

            logger.debug(f"BEDROCK_EMBED: {len(embedding)} dims in {latency:.0f}ms")
            return embedding

        except Exception as e:
            logger.error(f"BEDROCK_EMBED: Error - {e}")
            return []

    def _invoke(
        self,
        model_id: str,
        body: Dict,
        model_key: str,
        is_nova: bool = False
    ) -> BedrockResponse:
        """Internal invoke with timing and error handling."""
        start = time.time()

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            latency = (time.time() - start) * 1000

            # Extract text based on model type
            if is_nova:
                text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
                input_tokens = result.get('usage', {}).get('inputTokens', 0)
                output_tokens = result.get('usage', {}).get('outputTokens', 0)
            else:
                # Claude format
                text = result.get('content', [{}])[0].get('text', '')
                input_tokens = result.get('usage', {}).get('input_tokens', 0)
                output_tokens = result.get('usage', {}).get('output_tokens', 0)

            self._track_usage(model_key, latency, input_tokens, output_tokens)

            logger.debug(f"BEDROCK_{model_key.upper()}: {output_tokens} tokens in {latency:.0f}ms")

            return BedrockResponse(
                text=text.strip(),
                model_id=model_id,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"BEDROCK_{model_key.upper()}: Error - {e}")

            return BedrockResponse(
                text='',
                model_id=model_id,
                latency_ms=latency,
                success=False,
                error=str(e)
            )

    def _track_usage(
        self,
        model_key: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Track usage statistics."""
        self._call_count += 1
        self._total_latency_ms += latency_ms

        # Calculate cost
        costs = self.COSTS.get(model_key, (0, 0))
        cost = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000
        self._total_cost += cost

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'call_count': self._call_count,
            'total_latency_ms': self._total_latency_ms,
            'avg_latency_ms': self._total_latency_ms / max(self._call_count, 1),
            'total_cost_usd': self._total_cost
        }


# Singleton instance
_bedrock_client: Optional[BedrockClient] = None


def get_bedrock_client() -> BedrockClient:
    """Get the singleton BedrockClient instance."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    return _bedrock_client
