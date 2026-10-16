#!/usr/bin/env python3
"""
test_inference.py
Testa as predições de intenção do motor Laya / BGE-M3.
Pode ser executado localmente via Python ou remotamente via HTTP (FastAPI).
"""

import argparse
import json
import requests
import sys

SAMPLE_CRITERIA = {
    "clima": "Consultar previsão do tempo, clima e meteorologia",
    "fazenda": "Visualiza e gerencia a fazenda virtual no Jardim das Sombras",
    "faz-colher": "Colhe todas as plantas maduras nos canteiros da sua fazenda",
    "pesca": "Minigame de pescaria do bot Ravena, pescar peixes e ranking",
    "roleta": "Jogo da Roleta Russa no grupo",
    "cantada": "Envia cantadas engraçadas e espirituosas para descontrair",
    "dolar": "Consulta a cotação do dólar e câmbio",
    "ajuda": "Ajuda e suporte sobre os comandos do bot Ravena",
    "conversa": "Conversa casual, perguntas normais, bate-papo, dúvidas gerais sem intenção de executar comando do bot"
}

TEST_CASES = [
    {
        "query": "mas e se eu quiser criar boi na fazenda?",
        "expected": "conversa",
        "description": "Pergunta casual sobre boi na fazenda (NÃO deve chamar !fazenda)"
    },
    {
        "query": "o clima entre a diretoria e os funcionários azedou de vez",
        "expected": "conversa",
        "description": "Expressão figurada com a palavra 'clima' (NÃO deve chamar !clima)"
    },
    {
        "query": "sonhei que eu estava pescando peixe dourado num lago calmo",
        "expected": "conversa",
        "description": "Relato de sonho com pesca (NÃO deve chamar !pesca)"
    },
    {
        "query": "qual a previsão do tempo pra são paulo amanhã?",
        "expected": "clima",
        "description": "Invocação legítima de comando meteorológico"
    },
    {
        "query": "colher a soja da minha fazenda",
        "expected": "faz-colher",
        "description": "Invocação legítima de colheita da fazenda"
    },
    {
        "query": "mostrar minha fazenda",
        "expected": "fazenda",
        "description": "Invocação legítima de status geral da fazenda"
    },
    {
        "query": "manda uma cantada para mandar pra cremosa",
        "expected": "cantada",
        "description": "Invocação legítima de cantada"
    },
    {
        "query": "como funciona a pescaria no bot?",
        "expected": "pesca",  # ou 'ajuda'
        "description": "Dúvida sobre comando de pescaria"
    }
]


def test_via_http(endpoint_url, model_name="ravena-bgem3"):
    print(f"[test_inference] Testando via HTTP em: {endpoint_url} (model={model_name})\n")
    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        payload = {
            "state": tc["query"],
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Determine se a mensagem é para executar um comando do bot ou apenas conversa.",
                    "criteria": SAMPLE_CRITERIA
                }
            },
            "model": model_name
        }

        try:
            res = requests.post(endpoint_url, json=payload, timeout=10)
            if res.status_code != 200:
                print(f"[❌ ERRO {res.status_code}] {tc['query']} -> {res.text}")
                continue

            data = res.json()
            intent_res = data.get("answers", {}).get("intent", {})
            choice = intent_res.get("choice")
            conf = intent_res.get("confidence", 0.0)
            probs = intent_res.get("probabilities", {})

            is_ok = (choice == tc["expected"]) or (tc["expected"] == "pesca" and choice in ["pesca", "ajuda"])
            status = "✅ OK" if is_ok else "❌ FALHA"
            if is_ok:
                passed += 1

            print(f"[{status}] Query: \"{tc['query']}\"")
            print(f"       -> Predito: {choice} (Confiança: {conf * 100:.1f}%) | Esperado: {tc['expected']}")
            print(f"       -> Probabilidades top-2: {dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2])}")
            print()

        except Exception as e:
            print(f"[❌ EXCEÇÃO] {tc['query']} -> {e}")

    print("=" * 60)
    print(f"Resultado Final: {passed}/{total} testes passaram ({passed / total * 100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Testador de inferência Laya / BGE-M3")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8002/predict", help="URL do endpoint /predict")
    parser.add_argument("--model", type=str, default="ravena-bgem3", help="Nome do modelo a solicitar")
    args = parser.parse_args()

    test_via_http(args.url, model_name=args.model)


if __name__ == "__main__":
    main()
