# Checklist de entrega acadêmica

Estado em `reports/experiments/academic_delivery_results.json`.

```
academic_submission_ready = false
```

A entrega **não** está pronta. Tudo que podia ser produzido dentro do repositório
está produzido e validado; o que falta depende de ações que só uma pessoa pode
executar.

---

## PARTE 1 — CONCLUÍDO NO REPOSITÓRIO

### Notebook acadêmico

- [x] `notebooks/02_academic_delivery.ipynb` criado — 58 células (36 markdown, 22 código)
- [x] Narrativa em português, código com identificadores em inglês
- [x] 23 seções, do contexto do problema à conclusão
- [x] Executado de ponta a ponta localmente por `nbconvert` — **22/22 células, 0 falhas**
- [x] Sem caminho absoluto, sem `from churn`, sem upload local, sem credencial
- [x] Dataset baixado de fonte pública sem autenticação
- [x] SHA-256 do dataset verificado, com aborto em divergência
- [x] Partição reproduzida e conferida por digest de identificadores
- [x] Limiar reproduzido em precisão total de ponto flutuante
- [x] Métricas do holdout e matriz de confusão reproduzidas
- [x] Sem outputs no arquivo versionado (convenção do repositório)

### Relatório acadêmico

- [x] `reports/academic/academic_report.md` criado, em português
- [x] As 11 seções obrigatórias presentes
- [x] Seções adicionais: limitações, demonstração, reprodutibilidade, rastreabilidade, links
- [x] 8 figuras versionadas referenciadas sem duplicação de bytes
- [x] Tabela de rastreabilidade mapeando cada afirmação ao artefato de origem
- [x] Caveat de exposição do analista presente
- [x] Acurácia tratada como auxiliar em toda parte
- [x] Nenhum número fabricado
- [x] `reports/academic/academic_report.html` gerado (pronto para impressão)
- [x] `reports/academic/academic_report_DRAFT.pdf` gerado e inspecionado

### Material de apresentação

- [x] `academic/video_script.md` — roteiro completo, 13 blocos
- [x] `academic/presentation_outline.md` — 25 telas, todas mapeadas a artefatos existentes
- [x] `academic/colab_instructions.md` — passo a passo de execução e publicação
- [x] `academic/README.md` — índice do pacote

### Registro e verificação

- [x] `reports/experiments/academic_delivery_results.json` criado
- [x] `scripts/build_academic_record.py --verify` disponível e passando
- [x] `tests/test_academic_delivery.py` — 20 gates de conteúdo e honestidade
- [x] Artefatos das Fases 1–13 byte-idênticos
- [x] Nenhuma dependência nova; locais de root e serving inalterados

---

## PARTE 2 — EXTERNAL ACTION REQUIRED

Nenhum item abaixo pode ser executado automaticamente. Nenhum link foi inventado.

### Google Colab

- [ ] **Abrir** `notebooks/02_academic_delivery.ipynb` no Google Colab
      *(upload direto — o repositório ainda não é público)*
- [ ] **Executar tudo** (*Runtime → Run all*)
- [ ] **Validar os outputs** contra a lista de conferência em
      `academic/colab_instructions.md`, Passo 3
- [ ] **Compartilhar** com acesso "qualquer pessoa com o link", papel Leitor
- [ ] **Validar o link em janela anônima** — deve abrir sem pedir login
- [ ] **Registrar o link real** em `colab_shared_link` e no relatório
- [ ] **Exportar o notebook executado em PDF** (Arquivo → Imprimir → Salvar como PDF,
      com gráficos de segundo plano)
- [ ] **Inspecionar o PDF**: gráficos legíveis, tabelas não cortadas, sem página em
      branco, sem caminho local, sem marcador pendente

```
actual_google_colab_execution   = false
colab_shared_link               = null
colab_shared_link_verified      = false
actual_colab_export_pdf_present = false
```

### Vídeo de apresentação

- [ ] **Gravar** a apresentação seguindo `academic/video_script.md`
- [ ] **Publicar** em plataforma de livre acesso
      *(Drive com permissão restrita não atende ao requisito)*
- [ ] **Validar o link em janela anônima**
- [ ] **Registrar o link real** em `video_link` e no relatório

```
video_link          = null
video_link_verified = false
```

### Metadados de capa

Nenhum destes campos consta em qualquer arquivo do repositório. Não foram inventados.

- [ ] Instituição de ensino
- [ ] Curso / programa de pós-graduação
- [ ] Disciplina
- [ ] Professor(a) responsável
- [ ] Cidade
- [ ] Data de entrega

*(Autor já preenchido: Vinicius Gomes — comprovado em `pyproject.toml` e no histórico
de commits.)*

### Fechamento do relatório

- [ ] Substituir os seis marcadores `[[EXTERNAL_INPUT_REQUIRED: …]]` da capa
- [ ] Preencher a seção *Links da entrega* com as URLs reais
- [ ] Remover o aviso **DRAFT** do topo
- [ ] Regenerar HTML e PDF (`uv run python scripts/render_academic_report.py`)
- [ ] Renomear o PDF de `academic_report_DRAFT.pdf` para `academic_report.pdf`
- [ ] Atualizar o registro; só então `academic_submission_ready = true`

### Repositório público *(opcional, mas recomendado)*

O remoto configurado (`Novachrono117/ML-Crunch`) responde **HTTP 404** sem
autenticação — é privado — e o commit atual não está publicado nele.

- [ ] Decidir se o repositório será tornado público
- [ ] Publicar os commits, se sim
- [ ] Atualizar o relatório com a URL verificada

---

## Ordem recomendada

```
1. preencher os metadados de capa
2. executar no Colab e validar
3. compartilhar e validar o link em janela anônima
4. exportar o PDF do notebook executado
5. gravar e publicar o vídeo; validar o link
6. preencher os links no relatório e remover o aviso DRAFT
7. regenerar HTML e PDF finais
8. atualizar o registro e rodar os testes
```

O passo 1 vem primeiro porque os campos de capa aparecem no PDF do relatório; deixá-lo
para o fim obriga a regerar tudo de novo.
