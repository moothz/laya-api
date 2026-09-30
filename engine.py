"""Laya API Engine: Optimized Text (Laya) & Vision (SigLIP/CLIP) Decision Engines."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger("laya.engine")


class LayaTextEngine:
    """High-performance Laya text decision engine with VRAM optimization and CPU thread tuning."""

    def __init__(
        self,
        device: Optional[str] = None,
        preload_models: Optional[List[str]] = None,
        max_loaded: int = 1,
        torch_dtype: Optional[torch.dtype] = None,
        cpu_threads: int = 4,
    ):
        self.device_str = device or os.getenv("LAYA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.max_loaded = max_loaded
        self.cpu_threads = cpu_threads

        if self.device.type == "cpu":
            torch.set_num_threads(self.cpu_threads)
            logger.info(f"LayaTextEngine initialized on CPU with {self.cpu_threads} torch threads.")
        else:
            logger.info(f"LayaTextEngine initialized on GPU: {self.device}")

        # Determine precision for GPU
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device.type == "cuda":
            amp_env = os.getenv("LAYA_CUDA_AMP", "bf16").lower()
            if amp_env in ("fp16", "float16"):
                self.dtype = torch.float16
            else:
                self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Which models to preload
        if preload_models is None:
            raw_preload = os.getenv("LAYA_PRELOAD_MODELS", "multilingual")
            if raw_preload.lower() in ("all", "1", "true"):
                self.preload_models = ["multilingual", "english"]
            elif raw_preload.lower() in ("0", "false", "none"):
                self.preload_models = []
            else:
                self.preload_models = [m.strip() for m in raw_preload.split(",") if m.strip()]
        else:
            self.preload_models = preload_models

        # Import laya components
        import laya
        from laya import Router, Agent

        # Build router with custom parameters
        self.router = Router(
            preload=False,
            device=self.device_str,
            default=os.getenv("LAYA_DEFAULT_MODEL", "multilingual"),
            max_loaded=self.max_loaded,
        )

        # Preload and optimize models in VRAM
        if self.preload_models:
            for model_name in self.preload_models:
                try:
                    logger.info(f"Preloading text model '{model_name}' on {self.device_str}...")
                    agent = self.router.load(model_name)
                    if self.device.type == "cuda" and self.dtype in (torch.bfloat16, torch.float16):
                        # Convert model weights directly to half precision on GPU
                        agent.model.to(dtype=self.dtype)
                        logger.info(f"Model '{model_name}' weights cast to {self.dtype} on GPU.")
                except Exception as e:
                    logger.warning(f"Could not preload model '{model_name}': {e}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"LayaTextEngine startup VRAM: {allocated_mb:.2f} MB")

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute text classification / decision."""
        t0 = time.perf_counter()
        result = self.router.predict(
            state,
            questions,
            model=model,
            task=task,
            lang=lang,
        )
        t1 = time.perf_counter()
        if isinstance(result, dict) and "routing" in result:
            result["routing"]["engine_timing_ms"] = round((t1 - t0) * 1000, 2)
        return result

    def predict_batch(
        self,
        requests: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        sort_by_length: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute batch text classification."""
        shape: Dict[str, Any] = {}
        if batch_size is not None:
            shape["batch_size"] = batch_size
        if sort_by_length:
            shape["sort_by_length"] = True
        return list(self.router.predict_batch(requests, **shape))


class LayaVisionEngine:
    """Zero-shot visual question answering and image classification engine using SigLIP / CLIP."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device_str = device or os.getenv("LAYA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.model_id = model_id or os.getenv("LAYA_VISION_MODEL", "google/siglip-base-patch16-224")

        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device.type == "cuda":
            amp_env = os.getenv("LAYA_CUDA_AMP", "bf16").lower()
            if amp_env in ("fp16", "float16"):
                self.dtype = torch.float16
            else:
                self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        logger.info(f"Loading Vision Engine with model '{self.model_id}' on {self.device_str} ({self.dtype})...")
        
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype)
        self.model.to(self.device).eval()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"LayaVisionEngine ready. Cumulative VRAM: {allocated_mb:.2f} MB")

    @staticmethod
    def load_image(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load image from base64 string, URL, file path, bytes, or PIL.Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")

        if isinstance(image_input, str):
            image_str = image_input.strip()

            # Case 1: Base64 data URI or raw base64
            if image_str.startswith("data:image/") or ";base64," in image_str:
                base64_data = image_str.split(";base64,")[-1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Check if likely raw base64 without prefix (length > 100, no newlines/urls)
            if not image_str.startswith(("http://", "https://", "/")) and len(image_str) > 200:
                try:
                    image_bytes = base64.b64decode(image_str)
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    pass

            # Case 2: HTTP / HTTPS URL
            if image_str.startswith(("http://", "https://")):
                import urllib.request
                req = urllib.request.Request(image_str, headers={"User-Agent": "LayaVision/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    image_bytes = resp.read()
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Case 3: Local file path
            if os.path.exists(image_str):
                return Image.open(image_str).convert("RGB")

            raise ValueError(f"Unable to load image from input (not a valid URL, file path, or base64): {image_str[:50]}...")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _render_question_candidates(self, qid: str, qdef: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract candidate labels and prompt texts for a question.
        Returns: (labels, prompt_texts)
        """
        qtype = qdef.get("type", "choice")
        instructions = qdef.get("instructions", "")
        criteria = qdef.get("criteria")

        labels: List[str] = []
        prompt_texts: List[str] = []

        if qtype == "choice":
            if isinstance(criteria, dict):
                for label, desc in criteria.items():
                    labels.append(str(label))
                    desc_str = str(desc) if desc else str(label)
                    # Format candidate prompt for vision model
                    prompt = f"{instructions}: {desc_str}" if instructions else desc_str
                    prompt_texts.append(prompt)
            elif isinstance(criteria, list):
                for item in criteria:
                    labels.append(str(item))
                    prompt = f"{instructions}: {item}" if instructions else str(item)
                    prompt_texts.append(prompt)
            else:
                raise ValueError(f"Question '{qid}' of type 'choice' requires dictionary or list criteria.")

        elif qtype == "score":
            if isinstance(criteria, list):
                for i, level in enumerate(criteria):
                    labels.append(f"level_{i}")
                    level_desc = str(level)
                    prompt = f"{instructions}: {level_desc}" if instructions else level_desc
                    prompt_texts.append(prompt)
            else:
                raise ValueError(f"Question '{qid}' of type 'score' requires list criteria.")

        elif qtype == "noul":
            labels = ["false", "true"]
            if isinstance(criteria, dict):
                false_desc = criteria.get("false", "no, the statement does not apply")
                true_desc = criteria.get("true", "yes, the statement applies")
            else:
                false_desc = f"not {instructions}" if instructions else "false / no"
                true_desc = instructions if instructions else "true / yes"
            prompt_texts = [str(false_desc), str(true_desc)]

        else:
            raise ValueError(f"Unsupported question type '{qtype}' for vision classification.")

        return labels, prompt_texts

    def predict_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        questions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Classify image against structured question schemas."""
        t0 = time.perf_counter()
        image = self.load_image(image_input)

        results: Dict[str, Any] = {}

        for qid, qdef in questions.items():
            labels, candidate_texts = self._render_question_candidates(qid, qdef)
            if not candidate_texts:
                continue

            # Process image and texts
            inputs = self.processor(
                text=candidate_texts,
                images=image,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "logits_per_image"):
                    logits = outputs.logits_per_image[0]
                else:
                    # Generic image-text similarity fallback
                    img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    logits = (img_embeds @ txt_embeds.T)[0] * 100.0

                probs = torch.softmax(logits, dim=-1).cpu().float().numpy()

            # Format distribution
            prob_dict = {label: float(round(float(p), 4)) for label, p in zip(labels, probs)}
            best_idx = int(probs.argmax())
            best_label = labels[best_idx]
            confidence = float(round(float(probs[best_idx]), 4))

            qtype = qdef.get("type", "choice")
            ans_val: Any = best_label
            if qtype == "noul":
                ans_val = (best_label == "true")
            elif qtype == "score":
                ans_val = best_idx

            results[qid] = {
                "answer": ans_val,
                "confidence": confidence,
                "probabilities": prob_dict,
                "type": qtype,
            }

        t1 = time.perf_counter()
        timing_ms = round((t1 - t0) * 1000, 2)

        return {
            "results": results,
            "routing": {
                "model": self.model_id,
                "device": str(self.device),
                "timing_ms": timing_ms,
                "questions_evaluated": len(results),
            },
        }
