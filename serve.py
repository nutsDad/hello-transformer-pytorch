"""FastAPI service for batched decoder-only text generation."""

import argparse
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from translate import build_inference_model, generate_texts


class GenerateRequest(BaseModel):
    """Validated JSON body for the generation endpoint."""

    prompts: list[str] = Field(min_length=1)
    max_new_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, gt=0)
    top_k: Optional[int] = Field(default=None, ge=0)
    do_sample: Optional[bool] = None
    stop_token_ids: Optional[list[int]] = None
    seed: Optional[int] = None


def create_app(checkpoint_path=None):
    """Create an application with one model loaded for the process lifetime."""
    if config.model_architecture != "decoder_only":
        raise RuntimeError("The inference service requires model_architecture='decoder_only'.")
    model = build_inference_model(checkpoint_path)
    app = FastAPI(title="Hello Transformer PyTorch", version="1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "architecture": config.model_architecture,
            "device": str(config.device),
        }

    @app.post("/generate")
    def generate(request: GenerateRequest):
        try:
            texts = generate_texts(
                request.prompts,
                model,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                do_sample=request.do_sample,
                stop_token_ids=request.stop_token_ids,
                seed=request.seed,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"texts": texts, "count": len(texts)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Path to a decoder-only checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(arguments.checkpoint),
        host=arguments.host,
        port=arguments.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
