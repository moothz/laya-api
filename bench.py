#!/usr/bin/env python3
"""Benchmark Laya Engine API (CPU vs GPU, Text & Image Workloads)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# --------------------------------------------------------------------------- #
# Test Payloads
# --------------------------------------------------------------------------- #

TEXT_PAYLOAD = {
    "state": {"body": "We were billed twice for March. Please refund it today."},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "invoices, payments, refunds", "other": "everything else"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "critical"],
        },
    },
}

# Generate a small in-memory test image
def _generate_test_image_base64() -> str:
    img = Image.new("RGB", (224, 224), color=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


IMAGE_PAYLOAD = {
    "image_base64": _generate_test_image_base64(),
    "questions": {
        "document_type": {
            "type": "choice",
            "instructions": "What kind of image is this?",
            "criteria": {
                "red_graphic": "red solid shape or graphic",
                "invoice": "invoice or billing document",
                "photo": "natural photo of people or scenery",
            },
        },
        "is_blurry": {
            "type": "noul",
            "instructions": "Is this image blurry or low quality?",
        },
    },
}


def send_post(url: str, data_dict: Dict[str, Any], timeout: float = 15.0) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
    """Send a POST request and measure execution latency in ms."""
    data_bytes = json.dumps(data_dict).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "LayaBench/1.0"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            t1 = time.perf_counter()
            resp_json = json.loads(body.decode("utf-8"))
            return (t1 - t0) * 1000.0, True, resp_json
    except Exception as e:
        return 0.0, False, None


def run_suite(url: str, name: str, payload: Dict[str, Any], total_requests: int = 150, concurrency: int = 1) -> Optional[Dict[str, Any]]:
    latencies: List[float] = []
    failed = 0
    t_start = time.perf_counter()

    if concurrency == 1:
        for _ in range(total_requests):
            lat, ok, _ = send_post(url, payload)
            if ok:
                latencies.append(lat)
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_post, url, payload) for _ in range(total_requests)]
            for f in futures:
                lat, ok, _ = f.result()
                if ok:
                    latencies.append(lat)
                else:
                    failed += 1

    total_time = time.perf_counter() - t_start
    rps = len(latencies) / total_time if total_time > 0 else 0

    if not latencies:
        print(f"  [!] Todas as requisições falharam em {url}")
        return None

    latencies.sort()
    stats = {
        "name": name,
        "total_time": total_time,
        "rps": rps,
        "success": len(latencies),
        "failed": failed,
        "avg": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p50": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
    }

    print(f"  Throughput (RPS):   {stats['rps']:.2f} req/s")
    print(f"  Sucesso / Falhas:   {stats['success']} / {stats['failed']}")
    print(f"  Latência Média:     {stats['avg']:.2f} ms")
    print(f"  Latência p50:       {stats['p50']:.2f} ms")
    print(f"  Latência p95:       {stats['p95']:.2f} ms")
    print(f"  Latência p99:       {stats['p99']:.2f} ms")
    return stats


def run_target(host: str, port: int, label: str, test_image: bool = True) -> Optional[Dict[str, Any]]:
    predict_url = f"http://{host}:{port}/predict"
    image_url = f"http://{host}:{port}/predict/image"
    health_url = f"http://{host}:{port}/health"

    print(f"\n================================================================================")
    print(f" 🚀 Testando: {label} (http://{host}:{port})")
    print(f"================================================================================")

    # Health & Warm-up
    print("[*] Aquecendo motor (10 requisições)...")
    warm_ok = 0
    for _ in range(10):
        _, ok, _ = send_post(predict_url, TEXT_PAYLOAD)
        if ok:
            warm_ok += 1
    if warm_ok == 0:
        print(f"[!] Falha ao conectar ao endpoint {predict_url}. Verifique se o serviço está ativo.")
        return None

    # Fetch health stats
    health_data = {}
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            health_data = json.loads(resp.read().decode("utf-8"))
            vram = health_data.get("vram", {})
            if vram and "allocated_mb" in vram:
                print(f"[*] Status do Dispositivo: {vram.get('device', 'N/A')} | VRAM Alocada: {vram.get('allocated_mb')} MB (Reservada: {vram.get('reserved_mb')} MB)")
    except Exception:
        pass

    # 1. Sequencial
    print("\n--- 1. Texto: Latência Sequencial (Concorrência = 1, Requests = 150) ---")
    s1 = run_suite(predict_url, "Seq (C=1)", TEXT_PAYLOAD, total_requests=150, concurrency=1)

    # 2. Concorrente
    print("\n--- 2. Texto: Throughput Concorrente (Concorrência = 8, Requests = 300) ---")
    s8 = run_suite(predict_url, "Conc (C=8)", TEXT_PAYLOAD, total_requests=300, concurrency=8)

    # 3. Saturado
    print("\n--- 3. Texto: Throughput Saturado (Concorrência = 16, Requests = 600) ---")
    s16 = run_suite(predict_url, "Sat (C=16)", TEXT_PAYLOAD, total_requests=600, concurrency=16)

    img_stats = None
    if test_image:
        print("\n--- 4. Imagem: Classificação Zero-Shot (Concorrência = 4, Requests = 40) ---")
        img_stats = run_suite(image_url, "Image (C=4)", IMAGE_PAYLOAD, total_requests=40, concurrency=4)

    return {
        "label": label,
        "port": port,
        "health": health_data,
        "c1": s1,
        "c8": s8,
        "c16": s16,
        "img": img_stats,
    }


def print_comparison(r1: Optional[Dict[str, Any]], r2: Optional[Dict[str, Any]]) -> None:
    if not r1 or not r2:
        return
    print("\n" + "=" * 90)
    print(f" 📊 COMPARATIVO DE PERFORMANCE: {r1['label']} (:{r1['port']}) vs {r2['label']} (:{r2['port']})")
    print("=" * 90)

    headers = ["Cenário / Teste", "Métrica", f"{r1['label']}", f"{r2['label']}", "Diferença / Speedup"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<16} | {headers[3]:<16} | {headers[4]}")
    print("-" * 90)

    scenarios = [
        ("c1", "Texto (C=1)"),
        ("c8", "Texto (C=8)"),
        ("c16", "Texto (C=16)"),
        ("img", "Imagem (C=4)"),
    ]

    for key, name in scenarios:
        d1 = r1.get(key)
        d2 = r2.get(key)
        if not d1 or not d2:
            continue

        # p50
        diff_p50 = ((d2["p50"] - d1["p50"]) / d1["p50"]) * 100
        sign_p50 = "+" if diff_p50 > 0 else ""
        print(f"{name:<20} | {'p50 (ms)':<12} | {d1['p50']:>13.2f} ms | {d2['p50']:>13.2f} ms | {sign_p50}{diff_p50:.1f}%")

        # p95
        diff_p95 = ((d2["p95"] - d1["p95"]) / d1["p95"]) * 100
        sign_p95 = "+" if diff_p95 > 0 else ""
        print(f"{'':<20} | {'p95 (ms)':<12} | {d1['p95']:>13.2f} ms | {d2['p95']:>13.2f} ms | {sign_p95}{diff_p95:.1f}%")

        # RPS
        ratio_rps = d2["rps"] / d1["rps"] if d1["rps"] > 0 else 0
        print(f"{'':<20} | {'RPS (req/s)':<12} | {d1['rps']:>13.2f}    | {d2['rps']:>13.2f}    | {ratio_rps:.2f}x")
        print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Laya Engine (CPU vs GPU)")
    parser.add_argument("--host", default="127.0.0.1", help="Host do serviço (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Porta única para testar (ex: 8002 ou 8005)")
    parser.add_argument("--compare", action="store_true", help="Compara automaticamente CPU (8005) e GPU (8002)")
    parser.add_argument("--cpu-port", type=int, default=8005, help="Porta do Laya CPU (default: 8005)")
    parser.add_argument("--gpu-port", type=int, default=8002, help="Porta do Laya GPU (default: 8002)")
    args = parser.parse_args()

    if args.port:
        label = f"Laya-Port-{args.port}"
        run_target(args.host, args.port, label)
    else:
        print("[*] Executando modo comparativo: CPU (Porta 8005) vs GPU (Porta 8002)...")
        res_cpu = run_target(args.host, args.cpu_port, "Laya-CPU")
        res_gpu = run_target(args.host, args.gpu_port, "Laya-GPU")
        print_comparison(res_cpu, res_gpu)


if __name__ == "__main__":
    main()
