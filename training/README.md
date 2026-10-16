# Pipeline de Treinamento: Laya BAAI/bge-m3 (Ravena Decision Engine)

Este diretório contém o pipeline completo para treinar e avaliar o modelo de classificação de intenção e desambiguação de comandos da Ravena (**BAAI/bge-m3 Fine-Tuned**).

---

## Estrutura de Arquivos

* `commands_catalog.json`: Catálogo consolidado dos comandos da Ravena (321 comandos reais).
* `export_commands_catalog.js`: Script de extração do catálogo a partir da base do bot Ravena.
* `generate_dataset.py`: Gerador de triplets contrastivos com Hard Negatives e benchmark.
* `ravena_train_triplets.jsonl`: Dataset de treino contrastivo (~2.500 exemplos).
* `ravena_eval_benchmark.jsonl`: Suíte de validação e benchmark de casos de borda.
* `train_bge_m3.py`: Script de treinamento com `MultipleNegativesRankingLoss` e precisão mista `bfloat16`.
* `test_inference.py`: Script de validação remota e local contra a API.
* `LAYA_BGE_M3_MIGRATION_GUIDE.md`: Guia de arquitetura e diagnóstico de migração.

---

## Como Re-treinar o Modelo

1. **Parar serviços no PM2 temporariamente para liberar VRAM (~16 GB livres):**
   ```bash
   pm2 stop bonsai-api f5-tts-api laya-gpu
   ```

2. **(Opcional) Regerar datasets caso o catálogo tenha sido alterado:**
   ```bash
   python3 generate_dataset.py
   ```

3. **Executar o treinamento contrastivo (duração: ~40s na RTX 5090):**
   ```bash
   python3 train_bge_m3.py \
     --base-model BAAI/bge-m3 \
     --train-data ravena_train_triplets.jsonl \
     --eval-data ravena_eval_benchmark.jsonl \
     --output-dir /opt/models/ravena-bgem3 \
     --epochs 3 \
     --batch-size 32 \
     --lr 2e-5
   ```

4. **Reiniciar os serviços no PM2:**
   ```bash
   pm2 restart bonsai-api f5-tts-api laya-gpu
   ```

5. **Validar a inferência via HTTP:**
   ```bash
   python3 test_inference.py --url http://127.0.0.1:8002/predict --model ravena-bgem3
   ```
