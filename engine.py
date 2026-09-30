"""Laya API Engine: Optimized Text (Laya) & Vision/Video (SigLIP/CLIP) Decision Engines."""

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
        from laya import Router

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
    """Zero-shot visual and video question answering engine using SigLIP / CLIP with multi-frame batching."""

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

    @classmethod
    def _load_raw_bytes(cls, input_data: Union[str, bytes]) -> bytes:
        """Fetch or decode raw bytes from URL, base64, or local file."""
        if isinstance(input_data, bytes):
            return input_data

        input_str = input_data.strip()

        # Case 1: Base64
        if input_str.startswith("data:") or ";base64," in input_str:
            base64_data = input_str.split(";base64,")[-1]
            return base64.b64decode(base64_data)

        if not input_str.startswith(("http://", "https://", "/")) and len(input_str) > 200:
            try:
                return base64.b64decode(input_str)
            except Exception:
                pass

        # Case 2: HTTP / HTTPS URL
        if input_str.startswith(("http://", "https://")):
            import urllib.request
            req = urllib.request.Request(input_str, headers={"User-Agent": "LayaVision/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()

        # Case 3: Local file path
        if os.path.exists(input_str):
            with open(input_str, "rb") as f:
                return f.read()

        raise ValueError(f"Unable to load media source (not a valid URL, file path, or base64): {input_str[:50]}...")

    @classmethod
    def extract_frames(
        cls,
        media_input: Union[str, bytes, Image.Image, List[Any]],
        num_frames: int = 5,
    ) -> List[Image.Image]:
        """Extract up to num_frames PIL Images from a static image, animated WebP/GIF, MP4/WebM video, or list of images."""
        if isinstance(media_input, list):
            frames: List[Image.Image] = []
            for item in media_input[:num_frames]:
                sub_frames = cls.extract_frames(item, num_frames=1)
                frames.extend(sub_frames)
            return frames if frames else [Image.new("RGB", (224, 224), color="black")]

        if isinstance(media_input, Image.Image):
            # Check if animated WebP / GIF
            if getattr(media_input, "is_animated", False) and getattr(media_input, "n_frames", 1) > 1:
                total = media_input.n_frames
                step = total / num_frames
                frames = []
                for i in range(num_frames):
                    idx = min(int(i * step), total - 1)
                    media_input.seek(idx)
                    frames.append(media_input.convert("RGB"))
                return frames
            return [media_input.convert("RGB")]

        # Raw bytes or string
        raw_bytes = cls._load_raw_bytes(media_input)

        # First, try decoding with Pillow (covers JPEG, PNG, static/animated WebP, GIF)
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            if getattr(pil_img, "is_animated", False) and getattr(pil_img, "n_frames", 1) > 1:
                total = pil_img.n_frames
                step = total / num_frames
                frames = []
                for i in range(num_frames):
                    idx = min(int(i * step), total - 1)
                    pil_img.seek(idx)
                    frames.append(pil_img.convert("RGB"))
                return frames
            return [pil_img.convert("RGB")]
        except Exception:
            pass

        # If Pillow fails, decode as video (MP4, WebM, AVI, MOV) using PyAV
        try:
            import av
            container = av.open(io.BytesIO(raw_bytes))
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                raise ValueError("No video stream found in media container.")

            # Decode frames
            all_frames: List[Image.Image] = []
            for frame in container.decode(video=0):
                all_frames.append(frame.to_image().convert("RGB"))

            if not all_frames:
                raise ValueError("No frames could be decoded from video.")

            # Uniform sampling
            total = len(all_frames)
            if total <= num_frames:
                return all_frames

            step = total / num_frames
            sampled = [all_frames[min(int(i * step), total - 1)] for i in range(num_frames)]
            return sampled
        except Exception as e:
            raise ValueError(f"Failed to decode media as image or video: {e}")

    def _render_question_candidates(self, qid: str, qdef: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract candidate labels and prompt texts for a question."""
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

    def predict_media(
        self,
        media_input: Union[str, bytes, Image.Image, List[Any]],
        questions: Dict[str, Any],
        num_frames: int = 5,
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """Classify image, multi-image, animated WebP, or video against structured question schemas."""
        t0 = time.perf_counter()
        frames = self.extract_frames(media_input, num_frames=num_frames)

        results: Dict[str, Any] = {}

        for qid, qdef in questions.items():
            labels, candidate_texts = self._render_question_candidates(qid, qdef)
            if not candidate_texts:
                continue

            # Batch process all frames simultaneously
            inputs = self.processor(
                text=candidate_texts,
                images=frames,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "logits_per_image"):
                    logits = outputs.logits_per_image
                else:
                    img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    logits = (img_embeds @ txt_embeds.T) * 100.0

                # Softmax across labels per frame -> shape [num_frames, num_labels]
                probs_per_frame = torch.softmax(logits, dim=-1).cpu().float()

                # Multi-frame aggregation
                if aggregation.lower() == "max" or len(frames) == 1:
                    probs_agg, _ = probs_per_frame.max(dim=0)
                    # Normalize to sum to 1.0
                    probs_sum = probs_agg.sum()
                    if probs_sum > 0:
                        probs_agg = probs_agg / probs_sum
                else:
                    # Default: mean pooling
                    probs_agg = probs_per_frame.mean(dim=0)

                probs_np = probs_agg.numpy()

            # Format distribution
            prob_dict = {label: float(round(float(p), 4)) for label, p in zip(labels, probs_np)}
            best_idx = int(probs_np.argmax())
            best_label = labels[best_idx]
            confidence = float(round(float(probs_np[best_idx]), 4))

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
                "frames_evaluated": len(frames),
                "aggregation": aggregation,
                "questions_evaluated": len(results),
            },
        }

    # Backward compatibility alias
    def predict_image(
        self,
        image_input: Union[str, bytes, Image.Image, List[Any]],
        questions: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.predict_media(image_input, questions, num_frames=5, aggregation="mean")
