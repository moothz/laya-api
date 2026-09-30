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
| `MAX_UPLOAD_SIZE_MB` | Limite de tamanho máximo para imagens Base64 e uploads (MB). | `50` |
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

---

### 2. `POST /predict/image` (Classificação Zero-Shot de Imagem via JSON)

Classifica uma imagem a partir de **URL**, **Base64** ou **caminho local**.

**Opção A: Envio via URL:**

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

**Opção B: Envio via Base64 (Data URI ou raw string):**

```bash
curl -X POST http://localhost:8002/predict/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD...",
    "questions": {
      "categoria": {
        "type": "choice",
        "instructions": "Qual é o tema principal desta imagem?",
        "criteria": {
          "futebol": "jogadores em campo ou partida de futebol",
          "gato": "gato ou felino doméstico",
          "documento": "recibo ou documento impresso"
        }
      }
    }
  }'
```

**Opção C: Envio de Múltiplas Imagens (`images` array, até 5 imagens):**

```bash
curl -X POST http://localhost:8002/predict/image \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "https://example.com/frame1.jpg",
      "https://example.com/frame2.jpg",
      "https://example.com/frame3.jpg"
    ],
    "aggregation": "mean",
    "questions": {
      "tema": {
        "type": "choice",
        "instructions": "Qual é o tema principal?",
        "criteria": {"futebol": "futebol", "outros": "outros temas"}
      }
    }
  }'
```

---

### 3. `POST /predict/video` (Classificação de Vídeo MP4 / WebM / WebP Animado)

Extrai uniformemente **5 frames** (configurável via `num_frames`) e realiza inferência paralela em lote com agregação temporal (`mean` ou `max`):

```bash
curl -X POST http://localhost:8002/predict/video \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/lance_futebol.mp4",
    "num_frames": 5,
    "aggregation": "max",
    "questions": {
      "acontecimento": {
        "type": "choice",
        "instructions": "O que acontece neste vídeo?",
        "criteria": {
          "gol_ou_comemoracao": "gol marcado, jogadores comemorando ou bola na rede",
          "falta_ou_cartao": "falta, árbitro apitando ou cartão exibido",
          "jogo_normal": "troca de passes e movimentação normal"
        }
      }
    }
  }'
```

---

### 4. `POST /predict/image/upload` e `POST /predict/video/upload` (Upload Direto Form-Data)

Envie diretamente arquivos binários de imagem (JPEG, PNG, WebP) ou vídeo (MP4, WebM, MOV):

```bash
# Upload de Imagem ou WebP animado
curl -X POST http://localhost:8002/predict/image/upload \
  -F "file=@/caminho/minha_foto.webp" \
  -F 'questions={"categoria": {"type": "choice", "instructions": "Classifique", "criteria": {"doc": "documento", "pessoa": "pessoa"}}}'

# Upload de Vídeo MP4
curl -X POST http://localhost:8002/predict/video/upload \
  -F "file=@/caminho/meu_video.mp4" \
  -F "num_frames=5" \
  -F "aggregation=mean" \
  -F 'questions={"esporte": {"type": "choice", "instructions": "Qual esporte?", "criteria": {"futebol": "partida de futebol", "basquete": "basquete"}}}'
```

---

### 5. `GET /health` (Telemetria de VRAM e Saúde)

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
