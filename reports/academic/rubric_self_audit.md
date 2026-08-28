# Auditoria do pacote contra a rubrica

Mapeamento explícito de cada critério para a seção do relatório, a seção do notebook e
a evidência versionada que o sustenta.

> **Sobre a pontuação.** Apenas três critérios têm pontuação comprovada no material
> disponível neste repositório. Para os demais, a categoria existe mas o valor em
> pontos **não foi extraído de forma confiável de nenhuma fonte** — e não é inventado
> aqui.
>
> ```
> points_not_reliably_extracted_from_source = true   (critérios 4, 5 e 6)
> ```

---

## 1 · Notebook executado, organização, clareza e reprodutibilidade — **8 pontos**

| Aspecto | Onde | Evidência |
|---|---|---|
| Notebook executado | `notebooks/02_academic_delivery.ipynb` | 22/22 células de código executadas sem erro por `nbconvert`; execução no **Colab pendente** (ação humana) |
| Organização | 23 seções numeradas, narrativa linear | 58 células — 36 de markdown, 22 de código |
| Clareza | cada resultado é seguido de leitura interpretativa | markdown intercalado, nunca blocos de código isolados |
| Reprodutibilidade — dados | fonte pública sem credenciais, SHA-256 verificado | seção 3 do notebook; aborta em divergência |
| Reprodutibilidade — partição | digests dos identificadores conferidos | seção 6 do notebook |
| Reprodutibilidade — ambiente | versões impressas na execução | seção 2 do notebook |
| Reprodutibilidade — sem dependência local | zero caminhos absolutos, zero `from churn`, zero upload | verificado por `tests/test_academic_delivery.py` |
| Paridade com os artefatos | cada número comparado com o valor congelado, na tela | seções 6, 9, 12, 13 do notebook |

**Ponto forte:** o notebook não apenas roda — ele **prova** que reproduz o protocolo
congelado, comparando 3 métricas de modelo, o limiar em precisão total de ponto
flutuante, 6 métricas do holdout e a matriz de confusão contra os artefatos
versionados.

**Lacuna conhecida:** execução real no Google Colab (`EXTERNAL_ACTION_REQUIRED`).

---

## 2 · Pré-processamento e engenharia de atributos — **5 pontos**

| Aspecto | Relatório | Notebook | Evidência |
|---|---|---|---|
| Exclusão de `customerID` | *Pipeline de ML* | seção 4 | justificada: identificador, não preditor |
| Tratamento de `TotalCharges` | *Dataset utilizado* | seções 4 e 7 | regra do zero estrutural, com o caso de rejeição explícito |
| Escala | *Pipeline de ML* | seção 7 | `StandardScaler`, com justificativa ligada à penalização L2 |
| Codificação | *Pipeline de ML* | seção 7 | `OneHotEncoder(handle_unknown="ignore")`, decisão operacional justificada |
| Categorias não vistas | *Pipeline de ML*, *Monitoramento* | seções 7 e 19 | contribuem zero; contadas pelo monitoramento |
| Ausência de vazamento | *Metodologia adotada* | seção 7 | tabela "qual passo tem estado"; `Pipeline` reajustado por fold |
| Engenharia de atributos | *Pipeline de ML* | seção 10 | 5 grupos testados, com hipótese e delta pareado por fold |
| Critério de adoção | *Pipeline de ML* | seção 10 | benefício demonstrado suficiente para justificar complexidade |
| Decisão final | *Pipeline de ML* | seção 10 | nenhuma feature adotada; E3 mantido como sensibilidade |

**Ponto forte:** a história é contada com precisão. Engenharia de atributos **foi
feita**; o que não houve foi adoção. E1 é analisado como caso instrutivo — soma
determinística de indicadores que o modelo já recebia.

---

## 3 · Modelagem e validação — **7 pontos**

| Aspecto | Relatório | Notebook | Evidência |
|---|---|---|---|
| Divisão protegida | *Metodologia adotada* | seção 6 | estratificada, semente 42, digests conferidos |
| Baselines | *Resultados experimentais* | seção 8 | `DummyClassifier` + logística; F1 = 0 explícito |
| Comparação de modelos | *Resultados experimentais* | seção 9 | 3 famílias com hipótese; deltas pareados |
| Métrica primária | todo o relatório | seções 1 e 8 | Average Precision; acurácia sempre auxiliar |
| Tuning | *Resultados experimentais* | seção 11 | validação cruzada aninhada, 5 externos × 4 internos |
| Regra de elegibilidade | *Metodologia adotada* | seção 11 | fixada antes dos resultados; ≥ 4 de 5 folds |
| Calibração | *Resultados experimentais* | seção 11 | sigmoid e isotônica avaliadas; `NONE` adotado |
| Limiar | *Resultados experimentais* | seção 12 | `F1_MAXIMIZATION`, validado de forma aninhada |
| Avaliação final | *Resultados experimentais* | seção 13 | holdout aberto **uma vez** |
| Intervalos de confiança | *Análise das métricas* | seção 14 | bootstrap percentil, 2000 reamostragens |
| Generalização | *Análise das métricas* | seção 14 | CV → holdout como descrição, não teste |

**Ponto forte:** a decisão de **não** adotar o boosting ajustado apesar do maior AP
médio. Demonstra que a regra de elegibilidade governa a decisão, e não o maior número.

---

## 4 · Explicabilidade — `points_not_reliably_extracted_from_source = true`

| Aspecto | Relatório | Notebook | Evidência |
|---|---|---|---|
| Estratégia justificada | *Explicabilidade do modelo* | seção 16 | por que decomposição exata em vez de SHAP |
| Global | *Explicabilidade do modelo* | seção 16 | ranking de dispersão da contribuição, 19 features |
| Verificação da identidade | *Explicabilidade do modelo* | seção 16 | `intercepto + Σ contribuições = logit`, erro ≤ 1e-12 |
| Local | *Explicabilidade do modelo* | seção 18 | 2 clientes sintéticos, declarados antes da previsão |
| Dependências estruturais | *Explicabilidade do modelo* | seção 17 | 7 regras verificadas; bloco agregado por soma exata |
| Linguagem não causal | ambos | seções 16 e 18 | "não são efeitos causais nem recomendações" |
| Ressalva de parametrização | *Explicabilidade do modelo* | seção 16 | one-hot redundante com o intercepto |

**Ponto forte:** exemplo local **sintético e declarado antes**, evitando seleção
retrospectiva de um caso conveniente do holdout.

---

## 5 · Estratégia de monitoramento — `points_not_reliably_extracted_from_source = true`

| Aspecto | Relatório | Notebook | Evidência |
|---|---|---|---|
| Esquema e qualidade | *Monitoramento* § 1 | seção 19 | 7 contadores por janela |
| Valores ausentes/inválidos | *Monitoramento* § 1 | seção 19 | rejeitados, não imputados |
| Categorias não vistas | *Monitoramento* § 2 | seção 19 | taxa monitorada; rastreamento limitado a 1024 |
| Consistência estrutural | *Monitoramento* § 3 | seção 19 | 7 regras observadas, não impostas |
| Drift de features | *Monitoramento* § 4 | seção 19 | PSI (numérico), TVD (categórico) |
| Drift de predição | *Monitoramento* § 5 | seção 19 | PSI de score, taxa de positivos previstos |
| PSI não é teste | *Monitoramento* § 6 | seção 19 | `OPERATIONAL_MONITORING_POLICY`, sem p-valor |
| Desempenho com rótulos | *Monitoramento* § 7 | seção 19 | ausência declarada, não mascarada |
| Critérios de retreinamento | *Monitoramento* | seção 20 | escada drift → investigação → diagnóstico → candidato |
| Sem cadência arbitrária | *Monitoramento* | seção 20 | explicitamente recusada |
| Privacidade | *Monitoramento* § 8 | seção 19 | sem payloads, sem identificadores |

**Ponto forte:** a recusa explícita de "retreinar todo mês" e de retreinamento
automático, com a escada de decisão desenhada.

---

## 6 · Qualidade do relatório final — `points_not_reliably_extracted_from_source = true`

| Aspecto | Estado |
|---|---|
| Todas as 11 seções obrigatórias | **presentes** — capa, introdução, problema, dataset, metodologia, pipeline, resultados, análise das métricas, explicabilidade, monitoramento, conclusão |
| Seções adicionais | limitações, demonstração, reprodutibilidade, rastreabilidade, links |
| Idioma | português técnico |
| Figuras | 8 figuras versionadas, referenciadas sem duplicação de bytes |
| Rastreabilidade | tabela mapeando cada afirmação ao artefato de origem |
| Números fabricados | **nenhum** — todos rastreáveis a artefatos executados |
| Campos de capa | **pendentes** (`EXTERNAL_INPUT_REQUIRED`) — não inventados |
| PDF | rascunho gerado; final pendente dos campos de capa |

---

## Resumo de estado

| Critério | Pontos | Estado no repositório |
|---|---|---|
| Notebook executado, organização, clareza, reprodutibilidade | **8** | completo, exceto execução real no Colab |
| Pré-processamento e engenharia de atributos | **5** | completo |
| Modelagem e validação | **7** | completo |
| Explicabilidade | não extraído | completo |
| Estratégia de monitoramento | não extraído | completo |
| Qualidade do relatório final | não extraído | completo, exceto campos de capa e links |

### O que ainda depende de ação humana

1. Executar o notebook no Google Colab e confirmar ausência de erro.
2. Compartilhar o notebook e registrar o link real.
3. Exportar o notebook executado em PDF.
4. Gravar e publicar o vídeo; registrar o link real.
5. Preencher os campos de capa do relatório.
6. Regenerar o PDF final sem o aviso de rascunho.
