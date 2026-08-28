# Pacote de entrega acadêmica

Índice do material da Fase 14. Este diretório guarda **instruções e roteiros**; os
documentos e registros ficam em `reports/academic/` e `reports/experiments/`.

```
academic_submission_ready = false
```

Tudo que pode ser produzido dentro do repositório está produzido e validado. O que
falta depende de ações humanas listadas em
[`reports/academic/submission_checklist.md`](../reports/academic/submission_checklist.md).

---

## O que existe

| Arquivo | Conteúdo |
|---|---|
| [`colab_instructions.md`](colab_instructions.md) | passo a passo para executar, compartilhar e exportar o notebook no Google Colab |
| [`video_script.md`](video_script.md) | roteiro falado da apresentação, 13 blocos |
| [`presentation_outline.md`](presentation_outline.md) | 25 telas, cada uma mapeada a uma figura ou gravação |

E fora deste diretório:

| Arquivo | Conteúdo |
|---|---|
| [`notebooks/02_academic_delivery.ipynb`](../notebooks/02_academic_delivery.ipynb) | narrativa acadêmica completa e auto-contida |
| [`reports/academic/academic_report.md`](../reports/academic/academic_report.md) | relatório com as 11 seções obrigatórias |
| `reports/academic/academic_report.html` | versão para impressão |
| `reports/academic/academic_report_DRAFT.pdf` | PDF de rascunho |
| [`reports/academic/rubric_self_audit.md`](../reports/academic/rubric_self_audit.md) | auditoria contra a rubrica |
| [`reports/academic/submission_checklist.md`](../reports/academic/submission_checklist.md) | o que está pronto e o que falta |
| `reports/experiments/academic_delivery_results.json` | registro determinístico da fase |

---

## O notebook, em uma frase

Uma reprodução **auto-contida** do protocolo congelado: baixa o dataset de fonte
pública sem credenciais, confere o SHA-256, refaz a partição, o pré-processamento, os
baselines, a comparação de modelos, o limiar e a avaliação do holdout — comparando
cada número, na tela, com o artefato versionado correspondente.

### Por que auto-contido, e não `pip install` do repositório

O repositório remoto configurado responde **HTTP 404** sem autenticação — é privado — e
o commit atual não está publicado nele. Um notebook que clonasse o repositório não
rodaria em um Colab limpo.

| | |
|---|---|
| **Repositório de engenharia** | fonte da verdade de produção: pipeline persistido, API, testes, monitoramento |
| **Notebook acadêmico** | reprodução auto-contida do protocolo congelado |

Ambos produzem os mesmos números — e o notebook prova isso a cada seção em vez de
afirmá-lo.

### Fonte dos dados

```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

Repositório público da IBM, HTTPS, **HTTP 200 sem cabeçalho de autorização**. O
conteúdo servido difere do arquivo versionado apenas nos terminadores de linha; após
normalização para a forma canônica `CRLF`, o SHA-256 é exatamente
`88be4b93…61d61358a`. O notebook **aborta** se não for.

---

## Estado da execução

| Item | Estado |
|---|---|
| Execução local completa (`nbconvert`) | **aprovada** — 22/22 células, 0 falhas |
| Execução real no Google Colab | **pendente** — ação humana |
| Link compartilhado do Colab | **null** — não inventado |
| PDF do notebook exportado do Colab | **ausente** — ação humana |
| Vídeo gravado e publicado | **pendente** — ação humana |
| Link do vídeo | **null** — não inventado |

> Execução local por `nbconvert` **não é** execução no Google Colab, e este pacote não
> a apresenta como tal.

---

## O que este pacote deliberadamente não faz

* **Não inventa link.** Nenhuma URL de Colab, YouTube, Drive ou Vimeo foi gerada.
* **Não inventa metadado de capa.** Instituição, curso, disciplina, docente, cidade e
  data não constam em nenhum arquivo do repositório e permanecem como
  `[[EXTERNAL_INPUT_REQUIRED: …]]`.
* **Não marca como pronto o que não está.** Todo campo `*_verified` continua `false`
  até a verificação humana.
* **Não produz PDF "final".** Enquanto houver marcador pendente, o PDF se chama
  `academic_report_DRAFT.pdf`.
* **Não toma decisão de modelagem.** Nada foi selecionado, ajustado ou recalibrado
  nesta fase: `model_changed = false`, `selection_after_academic_reproduction = false`.

---

## Verificação

```bash
# registro determinístico da fase
uv run python scripts/build_academic_record.py --verify

# gates de conteúdo e honestidade
uv run pytest tests/test_academic_delivery.py -v

# reexecução do notebook em ambiente isolado
uv run python -m nbconvert --to notebook --execute \
  --output-dir <scratch> notebooks/02_academic_delivery.ipynb
```
