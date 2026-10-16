#!/usr/bin/env python3
"""
train_bge_m3.py
Script de fine-tuning do modelo BAAI/bge-m3 para classificação de intenção
e desambiguação de comandos da Ravena vs Conversa Geral (Hard Negatives).

Otimizado para NVIDIA RTX 5090 (32GB VRAM) com bfloat16 e aceleração FlashAttention/PyTorch 2.x.
Compatível com sentence-transformers v2.x e v3.x.
"""

import argparse
import json
import os
import sys
import time
import torch

try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader
except ImportError:
    print("[ERRO] sentence-transformers ou torch não instalados.")
    print("Instale com: pip install torch sentence-transformers datasets accelerate")
    sys.exit(1)

# Suporte opcional à API v3 do sentence-transformers
try:
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from datasets import Dataset
    HAS_ST_V3 = True
except ImportError:
    HAS_ST_V3 = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning BGE-M3 para Ravena")
    parser.add_argument("--base-model", type=str, default="BAAI/bge-m3", help="HuggingFace model ID ou caminho local")
    parser.add_argument("--train-data", type=str, default="ravena_train_triplets.jsonl", help="Dataset de triplets")
    parser.add_argument("--eval-data", type=str, default="ravena_eval_benchmark.jsonl", help="Dataset de benchmark")
    parser.add_argument("--output-dir", type=str, default="./model_output/ravena-bgem3", help="Diretório para salvar o modelo")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote (RTX 5090 suporta 32-64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Taxa de warmup")
    return parser.parse_args()


def load_triplets(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {file_path}")
    triplets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def load_eval_data(file_path):
    if not os.path.exists(file_path):
        return []
    benchmark = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                benchmark.append(json.loads(line))
    return benchmark


def evaluate_model(model, benchmark_data, device="cuda"):
    """Avalia o modelo contra o benchmark de casos reais e hard negatives."""
    if not benchmark_data:
        print("[evaluate_model] Nenhum benchmark fornecido. Pulando.")
        return

    # Tenta carregar catálogo para descrições dos comandos
    catalog_map = {}
    catalog_path = os.path.join(os.path.dirname(__file__), "commands_catalog.json")
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                cat = json.load(f)
                catalog_map = {c["name"]: c.get("desc", "") for c in cat}
        except Exception:
            pass

    catalog_map.setdefault("conversa", "Conversa casual, perguntas normais, bate-papo, dúvidas gerais sem intenção de executar comando do bot")
    catalog_map.setdefault("pesca", "Minigame de pescaria do bot Ravena, pescar peixes e ranking")
    catalog_map.setdefault("roleta", "Jogo da Roleta Russa no grupo")
    catalog_map.setdefault("dolar", "Consulta a cotação do dólar e câmbio")

    print("\n" + "="*70)
    print("  AVALIAÇÃO DO MODELO - BENCHMARK DE INTENÇÕES E HARD NEGATIVES")
    print("="*70)

    model.eval()
    correct = 0
    total = len(benchmark_data)
    hard_negatives_correct = 0
    hard_negatives_total = 0

    with torch.no_grad():
        for item in benchmark_data:
            query = item["query"]
            expected = item["expected"]
            candidates = item["candidates"]
            is_hn = "Hard negative" in item.get("note", "")

            # Formata candidatos como 'nome: desc' se disponíveis
            formatted_candidates = [
                f"{c}: {catalog_map[c]}" if c in catalog_map and catalog_map[c] else c
                for c in candidates
            ]
            q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            c_embs = model.encode(formatted_candidates, convert_to_tensor=True, normalize_embeddings=True)

            scores = torch.mv(c_embs, q_emb)
            best_idx = int(torch.argmax(scores).item())
            predicted = candidates[best_idx]

            is_correct = (predicted == expected)
            if is_correct:
                correct += 1
            if is_hn:
                hard_negatives_total += 1
                if is_correct:
                    hard_negatives_correct += 1

            status = "✅ OK" if is_correct else "❌ ERRO"
            print(f"[{status}] Query: '{query}' -> Predito: '{predicted}' (Esperado: '{expected}')")

    acc = (correct / total) * 100 if total > 0 else 0
    hn_acc = (hard_negatives_correct / hard_negatives_total) * 100 if hard_negatives_total > 0 else 0

    print("\n" + "-"*70)
    print(f"Acurácia Geral: {correct}/{total} ({acc:.2f}%)")
    print(f"Acurácia em Hard Negatives (Conversa vs Comandos): {hard_negatives_correct}/{hard_negatives_total} ({hn_acc:.2f}%)")
    print("="*70 + "\n")


def train_v3(args, model, triplets, benchmark_data):
    """Treinamento usando SentenceTransformerTrainer (sentence-transformers v3.x)."""
    print(f"[train_bge_m3] Executando pipeline com SentenceTransformerTrainer (v3)...")

    train_dict = {
        "anchor": [t["anchor"] for t in triplets],
        "positive": [t["positive"] for t in triplets],
        "negative": [t["negative"] for t in triplets]
    }
    train_dataset = Dataset.from_dict(train_dict)

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss
    )

    print(f"[train_bge_m3] Iniciando treinamento de {len(triplets)} triplets por {args.epochs} épocas...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train_bge_m3] Treinamento concluído em {elapsed:.1f}s!")

    print(f"[train_bge_m3] Salvando modelo final em: {args.output_dir}")
    model.save_pretrained(args.output_dir)


def train_v2(args, model, triplets, benchmark_data):
    """Treinamento clássico com DataLoader e MultipleNegativesRankingLoss (sentence-transformers v2.x)."""
    print(f"[train_bge_m3] Executando pipeline clássico com DataLoader (v2)...")

    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in triplets
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    print(f"[train_bge_m3] Iniciando model.fit: {len(train_examples)} exemplos, {args.epochs} épocas...")
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
        use_amp=True
    )
    elapsed = time.time() - t0
    print(f"[train_bge_m3] Treinamento concluído em {elapsed:.1f}s!")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("  TREINAMENTO BGE-M3 (RAVENA DECISION ENGINE)")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  bfloat16 suportado: {torch.cuda.is_bf16_supported()}")
    print("="*70)

    # 1. Carrega modelo base
    print(f"[train_bge_m3] Carregando modelo base: {args.base_model}...")
    model = SentenceTransformer(args.base_model, device=device)

    # 2. Carrega datasets
    triplets = load_triplets(args.train_data)
    benchmark_data = load_eval_data(args.eval_data)
    print(f"[train_bge_m3] Tripletas carregadas: {len(triplets)} | Benchmark: {len(benchmark_data)}")

    # 3. Avaliação Zero-Shot ANTES do treino
    print("\n--- AVALIAÇÃO PRÉ-TREINO (ZERO-SHOT) ---")
    evaluate_model(model, benchmark_data, device=device)

    # 4. Treinamento
    os.makedirs(args.output_dir, exist_ok=True)
    if HAS_ST_V3:
        train_v3(args, model, triplets, benchmark_data)
    else:
        train_v2(args, model, triplets, benchmark_data)

    # 5. Avaliação PÓS-TREINO
    print("\n--- AVALIAÇÃO PÓS-TREINO (FINE-TUNED) ---")
    evaluate_model(model, benchmark_data, device=device)

    print(f"[train_bge_m3] Tudo pronto! Modelo salvo em: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
