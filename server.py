"""Laya API Server: High-Performance Decision Engine for Text & Vision."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine import LayaTextEngine, LayaVisionEngine

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("laya.api")

# Global engine instances
text_engine: Optional[LayaTextEngine] = None
vision_engine: Optional[LayaVisionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_engine, vision_engine
    
    device = os.getenv("LAYA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    cpu_threads = int(os.getenv("LAYA_THREADS", "4"))
    enable_vision = os.getenv("LAYA_ENABLE_VISION", "1").lower() not in ("0", "false", "no")

    logger.info(f"Initializing Laya API Server on device: {device}...")

    # Initialize text engine
    text_engine = LayaTextEngine(
        device=device,
        cpu_threads=cpu_threads,
        max_loaded=int(os.getenv("LAYA_MAX_LOADED", "1")),
    )

    # Initialize vision engine if enabled
    if enable_vision:
        try:
            vision_engine = LayaVisionEngine(device=device)
        except Exception as e:
            logger.warning(f"Failed to initialize vision engine: {e}. Vision endpoints will be disabled.")
            vision_engine = None

    yield

    text_engine = None
    vision_engine = None


app = FastAPI(
    title="Laya Decision Engine API",
    version="1.0.0",
    description="API de alta performance para tomada de decisão e classificação de Texto (Laya) e Imagem (SigLIP Zero-Shot).",
    lifespan=lifespan,
)

# Enable CORS for all origins (accessible via VPN, local network, dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #

class Question(BaseModel):
    type: str = Field(..., description="Tipo da pergunta: choice | score | noul")
    instructions: str = Field(..., description="Prompt em linguagem natural ou instrução de classificação")
    criteria: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Critérios de classificação: dict de label -> descrição para 'choice', lista para 'score', opcional para 'noul'",
    )


class PredictRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]] = Field(
        ...,
        description="Texto ou registro estruturado a ser classificado. Se contiver 'image_url' ou 'image_base64', o motor de visão é utilizado automaticamente.",
    )
    questions: Dict[str, Question] = Field(..., min_length=1, description="Dicionário de perguntas")
    model: Optional[str] = Field(default=None, description="Override de modelo (ex: 'multilingual', 'english')")
    task: Optional[str] = Field(default=None, description="Identificador opcional de tarefa")
    lang: Optional[str] = Field(default=None, description="Código ISO do idioma (ex: 'pt', 'en') para ignorar autodetecção")


class BatchRequest(BaseModel):
    states: List[Union[str, Dict[str, Any], List[Any]]] = Field(..., min_length=1, max_length=64)
    questions: Dict[str, Question] = Field(..., min_length=1)
    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1)
    sort_by_length: bool = False


class PredictImageRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="URL pública da imagem (HTTP/HTTPS)")
    image_base64: Optional[str] = Field(default=None, description="String da imagem codificada em Base64 ou Data URI")
    image_path: Optional[str] = Field(default=None, description="Caminho local da imagem no servidor")
    state: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Metadados adicionais ou texto complementar")
    questions: Dict[str, Question] = Field(..., min_length=1, description="Dicionário de perguntas para a imagem")


def _questions_to_dict(questions: Dict[str, Question]) -> Dict[str, Any]:
    return {k: v.model_dump(exclude_none=True) for k, v in questions.items()}


# --------------------------------------------------------------------------- #
# API Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    """Health check com telemetria detalhada de memória VRAM e CPU."""
    vram_stats: Dict[str, Any] = {}
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        vram_stats = {
            "device": torch.cuda.get_device_name(0),
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_mb": round(total, 2),
            "free_approx_mb": round(total - reserved, 2),
        }

    return {
        "status": "ok" if text_engine is not None else "initializing",
        "device": str(text_engine.device) if text_engine else "unknown",
        "vram": vram_stats,
        "text_engine": {
            "ready": text_engine is not None,
            "preload_models": getattr(text_engine, "preload_models", []),
            "max_loaded": getattr(text_engine, "max_loaded", 1),
            "dtype": str(getattr(text_engine, "dtype", "unknown")),
        },
        "vision_engine": {
            "ready": vision_engine is not None,
            "model": getattr(vision_engine, "model_id", None),
            "dtype": str(getattr(vision_engine, "dtype", "unknown")),
        },
    }


@app.get("/models")
def list_models():
    """Lista modelos disponíveis de texto e visão."""
    return {
        "text_models": ["multilingual", "english", "typed-decisions"],
        "vision_models": ["google/siglip-base-patch16-224", "openai/clip-vit-base-patch32"],
        "default_text": os.getenv("LAYA_DEFAULT_MODEL", "multilingual"),
        "default_vision": getattr(vision_engine, "model_id", "google/siglip-base-patch16-224"),
    }


@app.get("/qtypes")
def list_qtypes():
    """Tipos de pergunta suportados."""
    return {"types": ["choice", "score", "noul"]}


@app.post("/predict")
def predict(req: PredictRequest):
    """Classificação de texto e multimodal."""
    if text_engine is None:
        raise HTTPException(status_code=503, detail="Serviço Laya ainda está inicializando.")

    # Check if state represents an image request
    if isinstance(req.state, dict) and vision_engine is not None:
        img_input = req.state.get("image_url") or req.state.get("image_base64") or req.state.get("image") or req.state.get("image_path")
        if img_input:
            try:
                return vision_engine.predict_image(img_input, _questions_to_dict(req.questions))
            except Exception as e:
                logger.exception("Falha na predição de imagem via /predict")
                raise HTTPException(status_code=400, detail=f"Falha no processamento de imagem: {str(e)}")

    try:
        return text_engine.predict(
            state=req.state,
            questions=_questions_to_dict(req.questions),
            model=req.model,
            task=req.task,
            lang=req.lang,
        )
    except Exception as e:
        logger.exception("Falha na predição de texto")
        raise HTTPException(status_code=500, detail=f"Erro durante a predição: {str(e)}")


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Classificação de texto em lote."""
    if text_engine is None:
        raise HTTPException(status_code=503, detail="Serviço Laya ainda está inicializando.")

    questions = _questions_to_dict(req.questions)
    controls = {k: v for k, v in [("model", req.model), ("task", req.task), ("lang", req.lang)] if v is not None}
    requests = [{"state": state, "questions": questions, **controls} for state in req.states]

    try:
        results = text_engine.predict_batch(
            requests=requests,
            batch_size=req.batch_size,
            sort_by_length=req.sort_by_length,
        )
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.exception("Falha no processamento em lote")
        raise HTTPException(status_code=500, detail=f"Erro durante predição em lote: {str(e)}")


@app.post("/predict/image")
def predict_image(req: PredictImageRequest):
    """Classificação Zero-Shot de imagens (URL, Base64 ou Caminho local)."""
    if vision_engine is None:
        raise HTTPException(status_code=503, detail="Motor de visão não está habilitado ou falhou na inicialização.")

    image_source = req.image_url or req.image_base64 or req.image_path
    if not image_source:
        raise HTTPException(
            status_code=422,
            detail="Informe ao menos uma fonte de imagem: 'image_url', 'image_base64' ou 'image_path'.",
        )

    try:
        return vision_engine.predict_image(
            image_input=image_source,
            questions=_questions_to_dict(req.questions),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Falha na classificação da imagem")
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")


MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@app.post("/predict/image/upload")
async def predict_image_upload(
    file: UploadFile = File(..., description="Arquivo de imagem (JPEG, PNG, WebP)"),
    questions: str = Form(..., description="JSON string contendo o dicionário de perguntas"),
):
    """Classificação de imagem enviada via Upload Multipart/Form-Data."""
    if vision_engine is None:
        raise HTTPException(status_code=503, detail="Motor de visão não está habilitado.")

    try:
        questions_dict = json.loads(questions)
    except Exception:
        raise HTTPException(status_code=422, detail="O campo 'questions' deve ser um JSON válido.")

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo excede o tamanho máximo permitido ({len(content)/(1024*1024):.1f}MB > {MAX_UPLOAD_SIZE_MB}MB).",
            )
        return vision_engine.predict_image(
            image_input=content,
            questions=questions_dict,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Falha no upload e predição de imagem")
        raise HTTPException(status_code=500, detail=f"Erro ao classificar upload: {str(e)}")
