#!/usr/bin/env python3
"""
generate_dataset.py
Gera dataset de treino contrastivo (triplets: anchor, positive, negative)
e dataset de avaliação para treinar o BGE-M3 para classificação de intenção
entre Comandos da Ravena e Conversa Geral (Hard Negatives).
"""

import json
import os
import random

SEED = 42
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "commands_catalog.json")
TRAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "ravena_train_triplets.jsonl")
EVAL_OUTPUT_PATH = os.path.join(BASE_DIR, "ravena_eval_benchmark.jsonl")

CONVERSA_LABEL = "conversa"
CONVERSA_DESC = "conversa: Conversa casual, perguntas normais, bate-papo, dúvidas gerais sem intenção de executar comando do bot"

# Modelos de frases de invocação imperativa para comandos
COMMAND_TEMPLATES = [
    "{cmd}",
    "{cmd} {arg}",
    "executar {name}",
    "roda o comando {name}",
    "usa {name}",
    "chama o {name}",
    "como faz pra usar o {name}?",
    "quero usar o {name}",
    "{name} agora por favor",
    "bot, {name}",
    "manda um {name}"
]

# Casos explícitos de Conversa Geral (sem nenhuma menção a comandos)
GENERAL_CONVERSATION_SAMPLES = [
    "olá, tudo bem com você?",
    "bom dia galera, como vocês estão?",
    "boa noite pessoal, até amanhã",
    "e aí, qual a boa de hoje?",
    "quem é você?",
    "o que você sabe fazer?",
    "qual o sentido da vida segundo a filosofia?",
    "como funciona a gravidade no espaço?",
    "quem foi o primeiro presidente do Brasil?",
    "qual a distância da Terra até o Sol?",
    "me explica a teoria da relatividade de Einstein",
    "o que você acha da evolução da inteligência artificial?",
    "você tem sentimentos ou consciência?",
    "qual o seu filme favorito de ficção científica?",
    "estou com sono e cansado do trabalho hoje",
    "me recomenda um livro interessante sobre história",
    "como fazer um café expresso perfeito?",
    "o que posso almoçar hoje? me dê sugestões",
    "como fazer um loop for em python?",
    "o que significa closure em javascript?",
    "como centralizar uma div na tela com flexbox?",
    "qual a diferença entre sql e nosql?",
    "me ajude a entender cálculo 1 derivadas",
    "como organizar meu tempo para estudar melhor?",
    "qual a montanha mais alta do mundo?",
    "quanto é 25 vezes 14?",
    "qual a capital do Canadá?",
    "você prefere gatos ou cachorros?",
    "estou um pouco triste hoje, me diga algo animador",
    "por que o céu é azul durante o dia?"
]

# HARD NEGATIVES CRÍTICOS: Frases conversacionais contendo palavras-chave de comandos,
# onde a intenção real é CONVERSA (e JAMAIS o comando do bot!)
HARD_NEGATIVE_CONVERSATIONS = [
    # !fazenda
    ("mas e se eu quiser criar boi na fazenda?", "fazenda"),
    ("meu avô tinha uma fazenda de café no interior de minas", "fazenda"),
    ("quanto custa o hectare de terra para montar uma fazenda?", "fazenda"),
    ("qual a diferença conceitual entre chácara, sítio e fazenda?", "fazenda"),
    ("será que compensa colocar placa de energia solar numa fazenda?", "fazenda"),
    ("o preço da soja caiu e os donos de fazenda estão preocupados", "fazenda"),

    # !clima
    ("o clima entre a diretoria e os funcionários azedou de vez", "clima"),
    ("o que você acha das mudanças climáticas globais?", "clima"),
    ("a reunião foi péssima, o clima ficou pesadíssimo", "clima"),
    ("adoro cidades com clima ameno de montanha", "clima"),
    ("essa discussão gerou um clima estranho no grupo", "clima"),

    # !pesca
    ("sonhei que eu estava pescando peixe dourado num lago calmo", "pesca"),
    ("qual o melhor tipo de isca para pesca esportiva de tucunaré?", "pesca"),
    ("meu pai foi pescar no rio Paraná no feriado", "pesca"),
    ("o setor de pesca artesanal sofreu com o derramamento de óleo", "pesca"),
    ("não como carne de gado nem frango, só como peixe", "pesca"),

    # !roleta
    ("ele foi imprudente e jogou uma verdadeira roleta russa na carreira", "roleta"),
    ("a roleta dos cassinos é matematicamente programada pra casa lucrar", "roleta"),
    ("a vida no trânsito das grandes cidades virou uma roleta russa", "roleta"),

    # !dolar / !euro
    ("por que os Estados Unidos conseguem emitir tanto dólar sem quebrar?", "dolar"),
    ("qual a história por trás da criação da moeda euro?", "dolar"),
    ("por que quase todo o comércio marítimo internacional usa o dólar?", "dolar"),

    # !ajuda (quando é dúvida de estudo/vida, e não sobre os comandos da Ravena)
    ("você pode me ajudar a escrever uma redação sobre meio ambiente?", "ajuda"),
    ("preciso de ajuda urgente para resolver essa equação de segundo grau", "ajuda"),
    ("me ajuda a escolher um presente de aniversário pra minha mãe", "ajuda"),
    ("como podemos ajudar pessoas em situação de rua no inverno?", "ajuda"),

    # !morreu
    ("meu peixinho dourado morreu hoje, fiquei muito chateado", "morreu"),
    ("em que ano morreu o físico Stephen Hawking?", "morreu"),
    ("qual filósofo clássico cunhou a frase 'Deus morreu'?", "morreu"),

    # !casar
    ("você acha que casamento antes dos 25 anos é arriscado?", "casar"),
    ("sonhei que estava na igreja indo casar de tênis e bermuda", "casar"),
    ("qual o custo médio de uma festa de casamento tradicional hoje?", "casar"),

    # !biscoito
    ("você prefere chamar de biscoito ou de bolacha?", "biscoito"),
    ("como fazer biscoito de polvilho assado que não murcha?", "biscoito"),

    # !horoscopo / !signo
    ("você acredita que os signos astrológicos influenciam a personalidade?", "horoscopo"),
    ("qual a origem histórica da astrologia e do zodíaco na babilônia?", "horoscopo"),

    # !piada
    ("a forma como aquela empresa lidou com a crise foi uma piada", "piada"),
    ("ele tentou se justificar na entrevista mas virou piada na internet", "piada"),

    # !noticias
    ("você tem alguma notícia do meu amigo que viajou?", "noticias"),
    ("estou sem internet faz três dias, perdi todas as notícias da semana", "noticias"),

    # !perfil
    ("qual o perfil psicológico ideal para trabalhar sob pressão?", "perfil"),
    ("gostei do perfil daquela empresa de tecnologia no LinkedIn", "perfil"),

    # !ranking
    ("qual o ranking dos países com maior índice de desenvolvimento humano?", "ranking"),
    ("o brasil subiu posições no ranking mundial de exportação de grãos", "ranking"),

    # !termo
    ("qual o termo médico técnico para perda repentina de olfato?", "termo"),
    ("não entendi bem aquele termo jurídico que o juiz citou no despacho", "termo")
]


def load_catalog():
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {CATALOG_PATH}. Execute export_commands_catalog.js primeiro.")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_triplets(catalog):
    triplets = []
    cmd_map = {c["name"]: c for c in catalog}
    # Aliases de alto uso no bot Ravena
    cmd_map["pesca"] = {"name": "pesca", "cmd": "!pesca", "desc": "Minigame de pescaria do bot Ravena, pescar peixes e ranking", "usage": ["!pesca"]}
    cmd_map["roleta"] = {"name": "roleta", "cmd": "!roleta", "desc": "Jogo da Roleta Russa no grupo", "usage": ["!roleta"]}
    cmd_map["dolar"] = {"name": "dolar", "cmd": "!dolar", "desc": "Consulta a cotação do dólar e câmbio", "usage": ["!dolar"]}
    all_cmd_names = list(cmd_map.keys())

    # 1. Triplet para cada comando do catálogo (Invocação positiva)
    for c in catalog:
        name = c["name"]
        cmd = c["cmd"]
        desc = c["desc"]
        usage = c["usage"]
        formatted_positive = f"{name}: {desc}"

        # Exemplos de uso como anchors positivos
        queries = []
        queries.append(cmd)
        queries.append(f"como usar o {cmd}")
        queries.append(f"quero executar o comando {name}")
        queries.append(f"roda o {name}")

        for u in usage:
            u_clean = u.strip()
            if u_clean:
                queries.append(u_clean)
                queries.append(u_clean.replace("!", ""))

        # Adiciona algumas variações de template
        arg_sample = "teste"
        if usage:
            parts = usage[0].split(maxsplit=1)
            if len(parts) > 1:
                arg_sample = parts[1]
        queries.append(f"manda um {name} {arg_sample}".strip())

        # Negativo para comando do catálogo: outro comando aleatório para contraste inter-comandos
        for q in set(queries):
            other_cmd_name = random.choice([k for k in all_cmd_names if k != name])
            other_cmd = cmd_map[other_cmd_name]
            negative = f"{other_cmd_name}: {other_cmd['desc']}"
            triplets.append({
                "anchor": q,
                "positive": formatted_positive,
                "negative": negative,
                "target_label": name
            })

    # 1.1 Invocação legítima de comandos em linguagem natural (perguntas diretas e ordens)
    LEGIT_COMMAND_INVOCATIONS = [
        ("qual a previsão do tempo pra são paulo amanhã?", "clima", "clima: Consultar previsão do tempo, clima e meteorologia"),
        ("previsão do tempo para amanhã", "clima", "clima: Consultar previsão do tempo, clima e meteorologia"),
        ("quantos graus está em curitiba?", "clima", "clima: Consultar previsão do tempo, clima e meteorologia"),
        ("vai chover hoje em são paulo?", "clima", "clima: Consultar previsão do tempo, clima e meteorologia"),
        ("como está o tempo hoje?", "clima", "clima: Consultar previsão do tempo, clima e meteorologia"),
        ("colher a soja da minha fazenda", "fazenda", "fazenda: Visualiza e gerencia a fazenda virtual no Jardim das Sombras, colheita e plantio"),
        ("colher soja da minha fazenda", "fazenda", "fazenda: Visualiza e gerencia a fazenda virtual no Jardim das Sombras, colheita e plantio"),
        ("como estão os animais da minha fazenda?", "fazenda", "fazenda: Visualiza e gerencia a fazenda virtual no Jardim das Sombras, colheita e plantio"),
        ("cuidar da plantação da fazenda", "fazenda", "fazenda: Visualiza e gerencia a fazenda virtual no Jardim das Sombras, colheita e plantio"),
        ("entrar na minha fazenda virtual", "fazenda", "fazenda: Visualiza e gerencia a fazenda virtual no Jardim das Sombras, colheita e plantio"),
        ("como funciona a pescaria no bot?", "pesca", "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"),
        ("jogar a vara de pesca no lago", "pesca", "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"),
        ("puxar o peixe da pescaria", "pesca", "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"),
        ("pescar peixes no rio", "pesca", "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"),
        ("quero pescar um peixe dourado", "pesca", "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"),
        ("manda uma cantada para mandar pra cremosa", "cantada", "cantada: Envia cantadas engraçadas e espirituosas para descontrair"),
        ("manda uma cantada para a cremosa", "cantada", "cantada: Envia cantadas engraçadas e espirituosas para descontrair"),
        ("me manda uma cantada agora", "cantada", "cantada: Envia cantadas engraçadas e espirituosas para descontrair"),
        ("manda cantada", "cantada", "cantada: Envia cantadas engraçadas e espirituosas para descontrair"),
        ("girar a roleta russa", "roleta", "roleta: Jogo da Roleta Russa no grupo"),
        ("jogar roleta russa no grupo", "roleta", "roleta: Jogo da Roleta Russa no grupo"),
        ("quanto está a cotação do dólar hoje?", "dolar", "dolar: Consulta a cotação do dólar e câmbio"),
        ("qual a cotação do dólar e do euro agora?", "dolar", "dolar: Consulta a cotação do dólar e câmbio"),
        ("quais são os comandos da ravena?", "ajuda", "ajuda: Ajuda e suporte sobre os comandos do bot Ravena"),
        ("como funciona o bot da ravena?", "ajuda", "ajuda: Ajuda e suporte sobre os comandos do bot Ravena"),
        ("ajuda sobre comandos do bot", "ajuda", "ajuda: Ajuda e suporte sobre os comandos do bot Ravena"),
        ("gera uma imagem de um gato cibernético", "imagine", "imagine: Cria e gera imagens usando inteligência artificial"),
    ]

    for legit_query, lbl, desc in LEGIT_COMMAND_INVOCATIONS:
        for _ in range(3):  # peso balanceado
            triplets.append({
                "anchor": legit_query,
                "positive": desc,
                "negative": CONVERSA_DESC,
                "target_label": lbl
            })

    # 2. Triplet para Conversa Geral (sem comando)
    for conv_query in GENERAL_CONVERSATION_SAMPLES:
        random_cmd_name = random.choice(all_cmd_names)
        random_cmd = cmd_map[random_cmd_name]
        formatted_negative = f"{random_cmd_name}: {random_cmd['desc']}"

        triplets.append({
            "anchor": conv_query,
            "positive": CONVERSA_DESC,
            "negative": formatted_negative,
            "target_label": CONVERSA_LABEL
        })

    # 3. Triplet para Hard Negatives Críticos (perguntas conversacionais com palavras de comandos)
    for conv_query, misleading_cmd_name in HARD_NEGATIVE_CONVERSATIONS:
        # Queremos que 'conversa' vença e que 'misleading_cmd' seja ativamente REPELIDO!
        misleading_desc = ""
        if misleading_cmd_name in cmd_map:
            misleading_desc = f"{misleading_cmd_name}: {cmd_map[misleading_cmd_name]['desc']}"
        elif misleading_cmd_name == "pesca":
            misleading_desc = "pesca: Minigame de pescaria do bot Ravena, pescar peixes e ranking"
        elif misleading_cmd_name == "roleta":
            misleading_desc = "roleta: Jogo da Roleta Russa no grupo"
        elif misleading_cmd_name == "dolar":
            misleading_desc = "dolar: Consulta a cotação do dólar e câmbio"
        else:
            misleading_desc = f"{misleading_cmd_name}: comando do bot"

        # Hard negatives balanceados (1 par limpo + 1 com prefixo)
        triplets.append({
            "anchor": conv_query,
            "positive": CONVERSA_DESC,
            "negative": misleading_desc,
            "target_label": CONVERSA_LABEL
        })
        triplets.append({
            "anchor": f"bot, {conv_query}",
            "positive": CONVERSA_DESC,
            "negative": misleading_desc,
            "target_label": CONVERSA_LABEL
        })

    random.shuffle(triplets)
    return triplets, cmd_map


def build_eval_benchmark(cmd_map):
    """Gera suíte de avaliação com os casos de borda mais difíceis."""
    benchmark = []

    # Casos de borda conversacionais que não podem disparar comando
    for conv_query, misleading_cmd_name in HARD_NEGATIVE_CONVERSATIONS:
        candidates = [misleading_cmd_name, CONVERSA_LABEL]
        # Adiciona mais 2 comandos aleatórios
        other_cmds = [k for k in cmd_map.keys() if k != misleading_cmd_name]
        candidates.extend(random.sample(other_cmds, min(2, len(other_cmds))))
        random.shuffle(candidates)

        benchmark.append({
            "query": conv_query,
            "expected": CONVERSA_LABEL,
            "candidates": candidates,
            "note": f"Hard negative para !{misleading_cmd_name}"
        })

    # Casos legítimos de invocação de comandos
    legit_invocations = [
        ("previsão do tempo para são paulo amanhã", "clima"),
        ("quantos graus está em curitiba?", "clima"),
        ("colher soja da minha fazenda", "fazenda"),
        ("como estão os animais da minha fazenda?", "fazenda"),
        ("jogar a vara de pesca no lago", "pesca"),
        ("puxar o peixe da pescaria", "pesca"),
        ("girar a roleta russa", "roleta"),
        ("quanto está a cotação do dólar hoje?", "dolar"),
        ("manda uma cantada para a cremosa", "cantada"),
        ("quais são os comandos da ravena?", "ajuda"),
        ("como funciona o bot da ravena?", "ajuda"),
        ("gera uma imagem de um gato cibernético", "imagine")
    ]

    for q, expected_cmd in legit_invocations:
        if expected_cmd in cmd_map:
            candidates = [expected_cmd, CONVERSA_LABEL]
            other_cmds = [k for k in cmd_map.keys() if k != expected_cmd]
            candidates.extend(random.sample(other_cmds, min(2, len(other_cmds))))
            random.shuffle(candidates)

            benchmark.append({
                "query": q,
                "expected": expected_cmd,
                "candidates": candidates,
                "note": f"Invocação legítima de !{expected_cmd}"
            })

    return benchmark


def main():
    print(f"[generate_dataset] Carregando catálogo de: {CATALOG_PATH}")
    catalog = load_catalog()
    print(f"[generate_dataset] Total de comandos carregados: {len(catalog)}")

    triplets, cmd_map = build_triplets(catalog)
    print(f"[generate_dataset] Total de triplets de treino gerados: {len(triplets)}")

    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[generate_dataset] Salvo: {TRAIN_OUTPUT_PATH}")

    benchmark = build_eval_benchmark(cmd_map)
    print(f"[generate_dataset] Total de testes de benchmark gerados: {len(benchmark)}")

    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for b in benchmark:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"[generate_dataset] Salvo: {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
