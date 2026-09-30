# ⚡ API de Decisão e Classificação com Laya (Texto + Imagem)

Uma API de tomada de decisão não-autorregressiva (System 1 Decision Engine) ultrarrápida e calibrada, construída com FastAPI, [Laya](https://github.com/convaiinnovations/laya) e [Google SigLIP](https://huggingface.co/google/siglip-base-patch16-224) para classificação de **Texto** e **Imagens (Zero-Shot)**.

Projetada para ser extremamente leve e eficiente, rodando localmente na sua infraestrutura ou servidor de rede/VPN com altíssimo throughput e latência na faixa de milissegundos.

---

## ✨ Recursos

- **Decisão Ultrarrápida em Texto**: Classificação de intenção, sentimento, triagem e roteamento via encoders calibrados Laya (ModernBERT / mmBERT).
- **Classificação Visual Zero-Shot**: Avaliação e classificação de imagens usando Google SigLIP / OpenAI CLIP diretamente a partir de instruções em linguagem natural.
- **Entrada Flexível de Imagem**: Suporte a URLs (`http/https`), strings codificadas em `base64` (ou data URI), caminhos locais no servidor ou upload direto multipart.
- **Otimização Extrema de VRAM (GPU)**: Roda em modo `bfloat16`/`float16` consumindo **menos de 1.2 GB de VRAM** (Texto + Visão simultaneamente).
- **Otimização Avançada para CPU (Zen 5 / Ryzen 9 9950X)**: Isolamento de afinidade no chiplet CCD0 (`taskset -c 0-7`), suporte nativo a AVX-512 BF16 e múltiplos workers Uvicorn contornando o GIL do Python.
- **Esquemas de Pergunta Laya**: Suporte completo a perguntas do tipo `choice` (múltipla escolha), `score` (escala numérica) e `noul` (booleano/binário).
- **Telemetria de VRAM e Hardware**: Endpoint `/health` expõe uso real de VRAM alocada, reservada e detalhes de execução.
- **Documentação Interativa (Swagger/OpenAPI)**: Interface gráfica em `/docs` para testes rápidos e exploração de payloads.

---

## ⚙️ Pré-requisitos

1. **Python 3.10+** (recomendado Python 3.11 ou 3.12/3.14).
2. **GPU NVIDIA (Opcional, para aceleração CUDA)**: Placas com suporte a CUDA 12+ (ex: RTX 30xx, 40xx, 50xx).
3. **PM2 (Opcional, para modo de produção)**: `npm install pm2 -g`.

---

## 🚀 Como Rodar

### 1. Clone ou Acesse o Repositório

```bash
cd /mnt/data/services/laya-api
```

### 2. Configure as Variáveis de Ambiente

Crie o arquivo `.env` a partir do modelo `.env.example`:

```bash
cp .env.example .env
```

### 3. Inicie o Servidor

#### Opção A: Execução Direta (Scripts)

- **Modo GPU (Porta 8002, VRAM < 1.5GB):**
  ```bash
  ./start-gpu.sh
  ```

- **Modo CPU (Porta 8005, 4 Workers CCD0):**
  ```bash
  ./start-cpu.sh
  ```

#### Opção B: Modo Produção (PM2)

O repositório já inclui o arquivo de configuração `ecosystem.config.js`:

```bash
# Iniciar serviço GPU
pm2 start ecosystem.config.js --only laya-gpu

# Iniciar serviço CPU
pm2 start ecosystem.config.js --only laya-cpu

# Salvar lista para reiniciar junto com o sistema operacional
pm2 save
```

Para verificar o status:
```bash
pm2 status
pm2 logs laya-gpu
```

---

## 📖 Documentação Interativa (Swagger)

A API conta com interface Swagger UI para visualização e testes interativos de todos os endpoints:

- **GPU**: `http://localhost:8002/docs`
- **CPU**: `http://localhost:8005/docs`
- **Acesso via VPN/Rede**: `http://192.168.195.x:8002/docs`

---

## ⚙️ Configuração (`.env`)

| Variável | Descrição | Padrão |
| :--- | :--- | :--- |
| `PORT` | Porta em que o servidor irá rodar. | `8002` |
| `HOST` | Endereço de bind de rede (`0.0.0.0` para toda a rede/VPN). | `0.0.0.0` |
| `LAYA_DEVICE` | Dispositivo de computação (`cuda` ou `cpu`). | `cuda` |
| `LAYA_CUDA_AMP` | Tipo de precisão em GPU (`bf16` ou `fp16`). | `bf16` |
| `LAYA_DEFAULT_MODEL` | Modelo padrão de texto Laya. | `multilingual` |
| `LAYA_PRELOAD_MODELS` | Modelos pré-carregados na inicialização. | `multilingual` |
| `LAYA_MAX_LOADED` | Limite de modelos de texto mantidos residentes em VRAM. | `1` |
| `LAYA_ENABLE_VISION` | Habilita motor de visão zero-shot (`1` ou `0`). | `1` |
| `LAYA_VISION_MODEL` | Modelo de visão Hugging Face. | `google/siglip-base-patch16-224` |
| `LAYA_THREADS` | Threads internas do PyTorch por worker na CPU. | `4` |
| `WORKERS` | Quantidade de processos worker Uvicorn (modo CPU). | `4` |
| `HF_HOME` | Diretório de cache de modelos Hugging Face. | `/mnt/data/services/laya/cache` |

---

## 📡 Endpoints da API

### 1. `POST /predict` (Texto ou Multimodal)

Classifica um texto ou objeto estruturado contra um conjunto de perguntas.

**Exemplo de Requisição (Texto):**

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"body": "Fui cobrado duas vezes no cartão de crédito em março. Gostaria do estorno."},
    "questions": {
      "departamento": {
        "type": "choice",
        "instructions": "Qual departamento deve atender esta solicitação?",
        "criteria": {
          "financeiro": "faturas, cobranças duplicadas, estornos e boletos",
          "suporte_tecnico": "problemas de conexão, erros de sistema e bugs",
          "outros": "outros assuntos gerais"
        }
      },
      "urgencia": {
        "type": "score",
        "instructions": "Quão urgente é esta solicitação?",
        "criteria": ["baixa", "média", "crítica"]
      }
    }
  }'
```

**Exemplo de Resposta:**

```json
{
  "answers": {
    "departamento": "financeiro",
    "urgencia": "crítica"
  },
  "confidence": {
    "departamento": 0.9821,
    "urgencia": 0.8954
  },
  "probabilities": {
    "departamento": {
      "financeiro": 0.9821,
      "suporte_tecnico": 0.0125,
      "outros": 0.0054
    },
    "urgencia": {
      "baixa": 0.0152,
      "média": 0.0894,
      "crítica": 0.8954
    }
  },
  "routing": {
    "model": "multilingual",
    "device": "cuda:0",
    "engine_timing_ms": 8.12
  }
}
```

---

### 2. `POST /predict/image` (Classificação Zero-Shot de Imagem)

Classifica uma imagem a partir de **URL**, **Base64** ou **caminho local**.

**Exemplo de Requisição (com URL ou Base64):**

```bash
curl -X POST http://localhost:8002/predict/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/ReceiptSwiss.jpg/800px-ReceiptSwiss.jpg",
    "questions": {
      "tipo_documento": {
        "type": "choice",
        "instructions": "Que tipo de imagem ou documento é este?",
        "criteria": {
          "recibo_fiscal": "recibo de compra, cupom fiscal ou nota fiscal",
          "documento_identidade": "carteira de identidade, CNH ou passaporte",
          "foto_natureza": "paisagem natural, pessoas ou animais"
        }
      },
      "legivel": {
        "type": "noul",
        "instructions": "O documento está legível e nítido?"
      }
    }
  }'
```

**Exemplo de Resposta:**

```json
{
  "results": {
    "tipo_documento": {
      "answer": "recibo_fiscal",
      "confidence": 0.9742,
      "probabilities": {
        "recibo_fiscal": 0.9742,
        "documento_identidade": 0.0185,
        "foto_natureza": 0.0073
      },
      "type": "choice"
    },
    "legivel": {
      "answer": true,
      "confidence": 0.9412,
      "probabilities": {
        "false": 0.0588,
        "true": 0.9412
      },
      "type": "noul"
    }
  },
  "routing": {
    "model": "google/siglip-base-patch16-224",
    "device": "cuda:0",
    "timing_ms": 7.84,
    "questions_evaluated": 2
  }
}
```

---

### 3. `POST /predict/image/upload` (Upload de Imagem Multipart)

Envie diretamente arquivos de imagem binários (PNG, JPEG, WebP):

```bash
curl -X POST http://localhost:8002/predict/image/upload \
  -F "file=@/caminho/minha_foto.jpg" \
  -F 'questions={"categoria": {"type": "choice", "instructions": "Classifique a imagem", "criteria": {"documento": "texto impresso", "pessoa": "rosto ou pessoa"}}}'
```

---

### 4. `GET /health` (Telemetria de VRAM e Saúde)

```bash
curl -s http://localhost:8002/health | python -m json.tool
```

**Resposta:**

```json
{
  "status": "ok",
  "device": "cuda",
  "vram": {
    "device": "NVIDIA GeForce RTX 5090",
    "allocated_mb": 1076.68,
    "reserved_mb": 1280.00,
    "total_mb": 32607.00,
    "free_approx_mb": 31327.00
  },
  "text_engine": {
    "ready": true,
    "preload_models": ["multilingual"],
    "max_loaded": 1,
    "dtype": "torch.bfloat16"
  },
  "vision_engine": {
    "ready": true,
    "model": "google/siglip-base-patch16-224",
    "dtype": "torch.bfloat16"
  }
}
```

---

## 📊 Benchmark e Teste de Performance

Para executar a bateria de testes e comparar a performance entre os modos CPU e GPU:

```bash
python bench.py --compare
```

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma **Issue** ou enviar um **Pull Request**.

---

## 📄 Licença

Este projeto está licenciado sob a Licença Apache 2.0 / MIT.
