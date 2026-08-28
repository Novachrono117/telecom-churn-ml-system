# Estrutura visual da apresentação

Roteiro falado: `academic/video_script.md`.

Este documento lista **o que precisa aparecer na tela** e de onde vem. Nada aqui exige
criar material novo: toda figura já existe e está versionada, e as telas de terminal e
demo são capturadas ao vivo durante a gravação.

> Não é necessário produzir um PowerPoint. Uma sequência de figuras versionadas mais
> gravação de tela cobre a apresentação inteira, e tem a vantagem de que cada imagem
> mostrada é rastreável a um artefato do repositório.

---

## Sequência de telas

| # | Seção do roteiro | O que mostrar | Origem |
|---|---|---|---|
| 1 | Abertura | Título, autor, uma linha de objetivo | slide simples |
| 2 | Problema | Distribuição do alvo — 73,5 % / 26,5 % | `reports/figures/eda/01_target_distribution.png` |
| 3 | Problema | Taxa de churn por contrato | `reports/figures/eda/04_churn_rate_by_contract.png` |
| 4 | Dataset | Tabela de estrutura (3 numéricas, 16 categóricas, 1 alvo) | relatório, seção *Dataset utilizado* |
| 5 | Dataset | As 11 linhas em branco de `TotalCharges` com `tenure = 0` ao lado | output do notebook |
| 6 | Pipeline | Diagrama do pipeline | relatório, seção *Pipeline de Machine Learning* |
| 7 | Pipeline | Tabela "qual passo tem estado e o que aprende" | relatório, mesma seção |
| 8 | Baselines | Tabela de baselines com **F1 = 0** destacado | relatório, *Resultados experimentais* |
| 9 | Comparação | Comparação de AP entre as três famílias | `reports/figures/model_comparison/01_cv_average_precision_comparison.png` |
| 10 | Comparação | Tabela de tuning, coluna "folds melhores" destacada | relatório, *Ajuste de hiperparâmetros* |
| 11 | Calibração | Diagrama de confiabilidade | `reports/figures/calibration/01_reliability_diagram.png` |
| 12 | Limiar | Precisão/recall × limiar, com 0,5 e 0,327 marcados | `reports/figures/threshold/01_precision_recall_vs_threshold.png` |
| 13 | Limiar | Tabela padrão × congelado (recall, precisão, taxa de positivos) | relatório, *Limiar de decisão* |
| 14 | Holdout | Curva Precision-Recall | `reports/figures/holdout/01_precision_recall_curve.png` |
| 15 | Holdout | Matriz de confusão | `reports/figures/holdout/03_confusion_matrix.png` |
| 16 | Holdout | Tabela de métricas com intervalos de confiança | relatório, *Análise das métricas* |
| 17 | Limitação | Texto do caveat de exposição do analista | resposta de `GET /api/v1/portfolio` |
| 18 | Explicabilidade | Dispersão da contribuição por feature | `reports/figures/interpretability/03_raw_feature_contribution_dispersion.png` |
| 19 | Explicabilidade | Explicação local de um cliente sintético | output do notebook |
| 20 | Explicabilidade | Bloco "sem internet": 7 linhas → 1 linha | output do notebook |
| 21 | Monitoramento | Tabela de métricas e limiares (PSI, TVD) | relatório, *Estratégia de monitoramento* |
| 22 | Monitoramento | Diagrama da escada de decisão de retreinamento | relatório, mesma seção |
| 23 | Demonstração | Gravação de tela: formulário → score → fatores | `/demo` ao vivo |
| 24 | Reprodutibilidade | Terminal: `--verify` retornando exit 0 | gravação de terminal |
| 25 | Conclusão | Modelo final, três métricas, limitações | slide de fechamento |

---

## Como levantar a demonstração para a tela 23

```bash
# Terminal 1 — sobe a API com a demo e o monitoramento ligados
CHURN_SERVING_PORTFOLIO_UI=1 CHURN_SERVING_MONITORING=1 \
  uv run --project serving python -m churn.serving
```

Depois abra `http://127.0.0.1:8000/demo` no navegador.

Na gravação, use o botão de exemplo pronto em vez de digitar 19 campos ao vivo —
mostra o mesmo e não desperdiça tempo de vídeo.

## Como gravar a tela 24

```bash
uv run python scripts/build_split.py --verify
uv run python scripts/freeze_model.py --verify
uv run python scripts/evaluate_holdout.py --verify
```

Mostrar os três retornando sem erro é suficiente; não é necessário rodar os dez.

---

## Recomendações de forma

* **Resolução:** 1920 × 1080. Aumente a fonte do terminal antes de gravar — texto de
  terminal em tamanho normal fica ilegível em vídeo comprimido.
* **Uma ideia por tela.** Se uma figura precisa de duas frases de contexto antes de
  fazer sentido, ela provavelmente deveria ser duas telas.
* **Destaque o número de que você está falando.** Um retângulo sobre a célula certa da
  tabela vale mais do que apontar com o cursor.
* **Sem animação de transição.** Elas custam tempo e não acrescentam informação.
* **Estilo acadêmico e sóbrio.** Sem estética de pitch de startup, sem "revolucionário",
  sem "IA de ponta".
