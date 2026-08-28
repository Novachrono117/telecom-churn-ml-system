# Executar e publicar o notebook no Google Colab

Este documento descreve as ações **humanas** necessárias para converter o notebook
versionado em uma entrega acadêmica completa. Nenhuma delas pode ser executada
automaticamente, e nenhum link produzido por elas existe ainda.

> **Estado atual:** `actual_google_colab_execution = false`.
> A execução local por `nbconvert` (validada, sem erros) **não é** execução no Google
> Colab e não pode ser apresentada como tal.

---

## Pré-requisitos

* Uma conta Google.
* O arquivo `notebooks/02_academic_delivery.ipynb` deste repositório.
* Conexão à internet no runtime do Colab — o notebook **baixa o dataset**, e essa é a
  única dependência externa que ele tem.

Não são necessários: conta no Kaggle, token de API, upload de CSV, montagem do Google
Drive, ou o repositório presente no filesystem do Colab.

---

## Passo 1 — Abrir o notebook no Colab

Escolha **uma** das opções:

**Opção A — upload direto (mais simples, funciona sempre)**

1. Acesse <https://colab.research.google.com>.
2. Menu **Arquivo → Fazer upload de notebook**.
3. Selecione `notebooks/02_academic_delivery.ipynb`.

**Opção B — a partir do GitHub**

Só funciona se o repositório for público. No momento ele **não é** (o remoto
configurado responde HTTP 404 sem autenticação), então use a Opção A até que isso
mude.

---

## Passo 2 — Executar tudo

1. Menu **Ambiente de execução → Executar tudo**
   (*Runtime → Run all*).
2. Aguarde. A execução local completa levou cerca de 15 segundos de CPU; no Colab,
   contando o download do dataset, espere algo entre 1 e 3 minutos.

### Passo 3 — Confirmar ausência de erro

Percorra o notebook de cima a baixo e confirme:

- [ ] **nenhuma** célula exibe traceback vermelho;
- [ ] a célula de aquisição de dados imprime `OK — integridade confirmada`;
- [ ] a célula de divisão imprime `treino OK` e `holdout OK`;
- [ ] a célula de limiar imprime `Identicos : True`;
- [ ] a tabela do holdout mostra diferenças na ordem de `1e-07` ou menores;
- [ ] a matriz de confusão reproduzida é `{'tn': 803, 'fp': 232, 'fn': 104, 'tp': 270}`;
- [ ] a célula final imprime `EXECUCAO CONCLUIDA` com todas as verificações positivas;
- [ ] todos os gráficos foram renderizados.

**Se qualquer verificação falhar, pare.** Não publique um notebook com célula
quebrada, e não edite os valores esperados para fazê-los coincidir — a divergência é
informação, não defeito a esconder.

Uma divergência plausível e aceitável: números levemente diferentes por causa de
outra versão do scikit-learn no Colab. O notebook **mostra** a diferença em vez de
escondê-la. Diferenças acima de `1e-04` merecem investigação antes de publicar.

---

## Passo 4 — Compartilhar o notebook

1. Botão **Compartilhar**, canto superior direito.
2. Em **Acesso geral**, selecione **Qualquer pessoa com o link**.
3. Papel: **Leitor** (*Viewer*). Não conceda edição.
4. **Copiar link**.

## Passo 5 — Validar o link em janela anônima

Este passo é o que separa "compartilhei" de "está compartilhado".

1. Abra uma **janela anônima / privada** do navegador.
2. Cole o link.
3. Confirme que a página abre **sem pedir login** e que os outputs executados
   aparecem.

Se pedir permissão, o compartilhamento não foi aplicado — volte ao Passo 4.

## Passo 6 — Registrar o link

Anote a URL. Ela precisa ser inserida em três lugares:

| Arquivo | Campo |
|---|---|
| `reports/experiments/academic_delivery_results.json` | `colab_shared_link` |
| `reports/academic/academic_report.md` | seção *Links da entrega* |
| `reports/academic/submission_checklist.md` | item correspondente |

---

## Passo 7 — Exportar o notebook executado em PDF

Com o notebook **já executado** e os outputs visíveis:

**Método recomendado — impressão do navegador**

1. Menu **Arquivo → Imprimir** (ou `Ctrl+P`).
2. Destino: **Salvar como PDF**.
3. Layout: **Retrato**; margens padrão; marque **Gráficos de segundo plano**
   (*Background graphics*) para preservar as cores dos gráficos.
4. Salve como `academic_notebook_colab.pdf`.

**Antes de considerar o PDF pronto, verifique:**

- [ ] o arquivo abre;
- [ ] todas as seções (1 a 23 e o apêndice) estão presentes;
- [ ] os gráficos aparecem e estão legíveis, não cortados;
- [ ] as tabelas não estão truncadas na lateral;
- [ ] não há página em branco acidental no meio do documento;
- [ ] nenhum caminho local do seu computador aparece no documento;
- [ ] nenhum marcador `[[EXTERNAL_INPUT_REQUIRED: …]]` restou.

> **Sobre células longas.** Se algum gráfico for cortado entre páginas, reduza a
> escala de impressão para 90 % e reimprima. Não recorte manualmente a imagem.

---

## Passo 8 — Gravar e publicar o vídeo

O roteiro completo está em `academic/video_script.md` e a estrutura visual em
`academic/presentation_outline.md`.

1. Grave a apresentação seguindo o roteiro.
2. Publique em uma plataforma de **livre acesso** (YouTube como "não listado" ou
   "público" atende; um Drive restrito **não** atende).
3. Valide o link em **janela anônima**, exatamente como no Passo 5.
4. Registre a URL nos mesmos três lugares do Passo 6, no campo `video_link`.

---

## Passo 9 — Fechar o relatório acadêmico

Depois que os links reais existirem:

1. Preencher os campos de capa do relatório
   (`[[EXTERNAL_INPUT_REQUIRED: …]]`): instituição, curso, disciplina, docente,
   cidade, data.
2. Preencher a seção *Links da entrega* com as URLs reais.
3. Remover o aviso **DRAFT** do topo do relatório.
4. Regenerar o HTML e o PDF do relatório
   (`uv run python scripts/render_academic_report.py`).
5. Atualizar `reports/experiments/academic_delivery_results.json` — os campos
   `*_verified` só passam a `true` depois da validação em janela anônima.

**Enquanto qualquer marcador `[[EXTERNAL_INPUT_REQUIRED: …]]` existir, nenhuma
renderização do relatório é versão final de entrega.**
