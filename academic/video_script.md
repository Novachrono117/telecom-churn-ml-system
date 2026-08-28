# Roteiro do vídeo de apresentação

> **Claude não grava nem publica vídeo.** Este é o roteiro; a gravação e a publicação
> são ações humanas. Enquanto não houver URL real, `video_link = null` e
> `video_link_verified = false`.

> **Sobre a duração.** O material da atividade disponível neste repositório **não
> especifica duração obrigatória**. Os tempos abaixo são
> `RECOMMENDED_TIMING_NOT_ASSIGNMENT_REQUIREMENT` — sugestões de ritmo, não exigências.
> Se a disciplina definir uma duração, ela prevalece: ajuste comprimindo as seções 4 e
> 10 primeiro, e **nunca** as seções 5, 7 e 11.

**Ritmo sugerido: ~12 minutos.**

---

## 1 · Abertura e problema — ~50 s

> "Este trabalho constrói um sistema de predição de churn para uma operadora de
> telecomunicações. Churn é o cliente que cancela — e em um negócio de assinatura, a
> receita não vem da venda, vem da permanência.
>
> A pergunta operacional é específica: **quais clientes estão em risco enquanto ainda
> é possível agir?** Uma lista de clientes perdidos tem valor analítico. Uma lista de
> clientes em risco tem valor operacional.
>
> E há uma armadilha logo na entrada. A base tem 26,5 % de churn. Um modelo que
> responde 'ninguém sai' para todo mundo acerta 73,5 % dos casos — e é completamente
> inútil. Por isso **acurácia não é o critério deste trabalho** em nenhum momento."

**Tela:** slide de título; gráfico da distribuição do alvo.

---

## 2 · Dataset — ~50 s

> "Telco Customer Churn, dado de exemplo da IBM: 7043 clientes, 19 preditores, alvo
> binário.
>
> O arquivo bruto é versionado e nunca editado à mão, e é identificado por SHA-256 —
> o notebook baixa os dados de uma fonte pública, calcula o hash e **aborta** se não
> bater. Não é possível rodar este trabalho sobre outra tabela sem perceber.
>
> Um detalhe de qualidade vale a menção: `TotalCharges` chega como texto, porque 11
> células estão em branco. Todas as 11 têm `tenure = 0` — são clientes que não
> fecharam o primeiro ciclo. O valor correto é zero. Mas a regra que adotamos não é
> 'preencher branco com zero': é preencher com zero **onde zero é justificável** e
> **falhar** onde não é. Branco com `tenure` positivo é erro, e continua sendo erro
> dentro da API."

**Tela:** tabela de estrutura; as 11 linhas em branco com `tenure = 0` ao lado.

---

## 3 · Pipeline e prevenção de vazamento — ~70 s

> "A primeira decisão irreversível é a divisão: 20 % estratificado por `Churn`,
> semente 42. Isso dá 5634 linhas de treino e 1409 de holdout.
>
> O holdout fica **lacrado**. Ele não participa de imputação, escala, codificação,
> seleção de features, comparação de modelos, tuning, calibração ou escolha de limiar.
>
> E essa proteção é **estrutural**, não disciplinar. Todo passo que aprende alguma
> coisa — o scaler, o encoder, o classificador — vive dentro de um `Pipeline` do
> scikit-learn. A validação cruzada reajusta o pipeline inteiro em cada fold, então
> nenhuma estatística atravessa a fronteira. Não depende de eu lembrar de fazer
> certo."

**Tela:** diagrama do pipeline; tabela "qual passo tem estado e o que aprende".

---

## 4 · Protocolo experimental e baselines — ~60 s

> "Todas as comparações usam a mesma partição de cinco folds, compartilhada entre os
> modelos — então as diferenças são pareadas: mesmos clientes, mesmos folds.
>
> Os baselines vêm primeiro, e o baseline de classe majoritária é o argumento mais
> claro deste trabalho: **73,5 % de acurácia com F1 igual a zero**. Ele nunca encontra
> um cliente em risco.
>
> A logística leva o Average Precision de 0,265 — que é a prevalência, o valor de um
> modelo sem informação — para 0,66."

**Tela:** tabela de baselines, com a linha do F1 = 0 destacada.

---

## 5 · Comparação de modelos e a decisão central — ~90 s

*Seção que não deve ser comprimida.*

> "Três famílias, cada uma com uma hipótese: logística como referência, random forest
> para interações, e histogram gradient boosting para o que um modelo linear deixa
> para trás.
>
> A logística vence: Average Precision de 0,661, contra 0,606 da random forest e
> 0,647 do boosting.
>
> Depois veio o tuning, com validação cruzada aninhada — o laço interno escolhe
> hiperparâmetros, o externo estima o desempenho do procedimento, e a escolha nunca
> vê as linhas em que é avaliada.
>
> E aqui está a decisão mais importante do trabalho. O boosting ajustado obteve o
> **maior** AP médio: 0,665 contra 0,661. E **não foi adotado**.
>
> A razão: a regra de elegibilidade foi fixada **antes** de eu ver os resultados, e
> exigia melhora em pelo menos 4 dos 5 folds. O boosting ganhou em 3 e perdeu em 2. A
> vantagem média vinha de um ganho grande em poucos folds — isso é variância, não
> superioridade.
>
> Adotá-lo seria escolher pelo maior número em vez de por evidência. O padrão
> pré-registrado é manter o modelo mais simples."

**Tela:** tabela de comparação; depois a tabela de tuning com a coluna "folds
melhores" em destaque.

---

## 6 · Calibração — ~40 s

> "Calibração foi **experimentada**, não ignorada. Sigmoid e isotônica, contra o
> modelo não calibrado, com Brier score como critério.
>
> O sigmoid melhorou o Brier em 1 fold de 5. A isotônica piorou a ordenação em 5 de 5.
> Nenhum dos dois cumpriu o protocolo de adoção.
>
> Resultado: `calibration_policy = NONE`. E a consequência precisa ser dita: os scores
> são **posições de ordenação**, não frequências. Uma probabilidade de 0,42 não
> significa '42 % destes clientes vão sair'."

**Tela:** diagrama de confiabilidade; tabela de calibração.

---

## 7 · O limiar — ~90 s

*Seção que não deve ser comprimida.*

> "O `predict` do scikit-learn usa 0,5. Esse número vem da implementação, não do
> problema.
>
> A política adotada é maximização de F1 sobre as probabilidades out-of-fold do
> treino — validada de forma aninhada, e melhorou o F1 em 5 de 5 folds. O limiar
> congelado é **0,327**.
>
> O que ele faz: o recall sobe de 0,54 para 0,74; a precisão cai de 0,65 para 0,56; e
> a operação passa a sinalizar 35 % da carteira em vez de 22 %.
>
> Agora, o ponto honesto. **Isto não é 'o limiar ótimo para o negócio'.** É o limiar
> que a política declarada seleciona com os dados que eu tenho. O limiar correto
> dependeria do custo de uma campanha de retenção, do valor de um cliente retido e da
> capacidade da equipe. Eu não tenho esses três números — e não inventei nenhum deles
> para produzir uma justificativa mais elegante."

**Tela:** curva precisão/recall × limiar, com as linhas de 0,5 e 0,327 marcadas.

---

## 8 · Resultado final no holdout — ~70 s

> "O holdout foi aberto **uma vez**, depois de tudo congelado. 1409 clientes.
>
> Average Precision 0,634, com intervalo de 0,578 a 0,686 — contra 0,265 sem
> informação. ROC-AUC 0,842. Recall 0,722, precisão 0,538, F1 0,616.
>
> Dos 374 clientes que realmente saíram, o modelo encontrou 270.
>
> Duas ressalvas de leitura. ROC-AUC de 0,842 **não é** '84 % de acerto' — é a
> probabilidade de ordenar corretamente um par. E a acurácia de 0,762 quase não
> informa, porque a classe majoritária já entrega 0,735.
>
> Comparando desenvolvimento com holdout: o AP caiu 0,028. Queda pequena, na direção
> esperada. É descrição, não teste — não calculei p-valor nenhum."

**Tela:** tabela de métricas com ICs; curva PR; matriz de confusão.

---

## 9 · A limitação que eu não escondo — ~50 s

> "Uma limitação real, e desconfortável.
>
> O holdout foi protegido corretamente a partir da divisão — nenhuma estatística foi
> ajustada nele. Mas a análise exploratória inicial do projeto foi feita sobre as 7043
> linhas, **antes** de o holdout ser separado. As hipóteses que orientaram o trabalho
> foram formadas olhando dados que incluíam o conjunto de teste.
>
> Então: **a estimativa final pode carregar um viés otimista que não é quantificável
> aqui.** Medi-lo exigiria um segundo conjunto de teste que não existe.
>
> Vazamento algorítmico não houve. Vazamento de conhecimento do analista houve, em
> grau desconhecido. Isso está no relatório e na API — é um campo do artefato, para
> que as métricas não possam ser exibidas sem ele."

**Tela:** o texto do caveat como aparece no `GET /api/v1/portfolio`.

---

## 10 · Explicabilidade — ~80 s

> "Não usei SHAP. E isso é uma escolha, não uma lacuna.
>
> SHAP existe para sondar modelos cuja superfície de resposta é desconhecida. Este
> modelo é conhecido em forma fechada: o logit é o intercepto mais a soma de 19
> contribuições. A decomposição é **exata** — verificada a 10⁻¹² em toda explicação
> servida. Um estimador por amostragem daria uma resposta pior a uma pergunta já
> respondida exatamente.
>
> Globalmente: `tenure` domina, seguido de `MonthlyCharges`, `InternetService`,
> `Contract` e `TotalCharges`. `gender` e `Partner` são praticamente inertes.
>
> E um detalhe técnico do qual me orgulho. Quando um cliente não tem internet, **sete
> colunas mudam juntas**. Não são sete evidências independentes — é **um fato**
> codificado sete vezes. Uma explicação que liste sete linhas sugere sete razões
> quando existe uma. Então elas são agregadas em um bloco, cujo valor é a soma exata
> das partes."

**Tela:** ranking de dispersão; explicação local de um cliente sintético; o bloco
"sem internet" antes e depois da agregação.

---

## 11 · Monitoramento e retreinamento — ~80 s

*Seção que não deve ser comprimida.*

> "O modelo congelado não se degrada sozinho — o mundo em volta muda.
>
> O desenho cobre qualidade de dados, categorias não vistas, consistência estrutural,
> drift de features com PSI e TVD, e drift de predição.
>
> Um ponto metodológico: **PSI e TVD não são testes estatísticos.** São distâncias
> descritivas. Os limiares de 0,10 e 0,25 são política operacional escolhida antes de
> existir dado de produção — não são níveis de significância, e cruzar um deles não é
> evidência de degradação.
>
> Desempenho só é medido **quando os rótulos chegam**. Não calculo acurácia em
> produção, porque não existe verdade de campo — churn só é observável depois de
> semanas. Declarar essa ausência é melhor do que mascará-la.
>
> E sobre retreinamento: **não prescrevo 'retreinar todo mês'**. Drift dispara
> **investigação**. Se a causa for técnica, corrige-se a origem. Só com rótulos e
> degradação corroborada é que se cogita um candidato — e o candidato refaz o
> protocolo inteiro, com holdout novo, antes de qualquer promoção. Não há
> retreinamento automático."

**Tela:** tabela de métricas e limiares; diagrama da escada de decisão.

---

## 12 · Demonstração — ~40 s

> "O modelo está empacotado: uma API com endpoints de predição, explicação, metadados
> e monitoramento, e uma página de demonstração com o formulário das 19 features.
>
> Aqui eu preencho um cliente, e recebo probabilidade, decisão e as contribuições que
> a produziram — a mesma decomposição exata da seção anterior.
>
> Mas quero ser claro: **a interface não é evidência de que o modelo está correto.**
> Essa evidência vem do protocolo experimental. Uma tela bonita com um modelo mal
> validado continua sendo um modelo mal validado."

**Tela:** gravação da demo — preencher o formulário, submeter, mostrar score,
decisão e fatores.

---

## 13 · Conclusão — ~50 s

> "Modelo final: regressão logística sobre as 19 features originais, sem calibração,
> com limiar 0,327.
>
> Escolhida porque venceu na métrica primária, porque nenhum procedimento de tuning
> cumpriu a regra de elegibilidade, e porque quando nada justifica a substituição, o
> padrão é ficar com o mais simples.
>
> No holdout: Average Precision 0,634, ROC-AUC 0,842, recall 0,722.
>
> E a afirmação final, com cuidado. Este trabalho demonstra um sistema reprodutível
> que **identifica** clientes em risco, validado sob um protocolo que protege o
> conjunto de teste, com previsões explicadas exatamente e monitoramento desenhado.
>
> Ele **não** demonstra que reduz churn. Isso exigiria um experimento de intervenção
> com grupo de controle, que não foi conduzido. Predizer quem sai e reduzir quantos
> saem são resultados diferentes — e eu medi apenas o primeiro."

**Tela:** slide de fechamento com o modelo final, as três métricas e as limitações.

---

## Notas de gravação

* **Tenha os números na tela.** Falar "melhorou bastante" é fraco; mostrar
  "0,661 contra 0,606, com 0 de 5 folds a favor" é forte.
* **Não apresse as seções 5, 7 e 11.** São elas que distinguem este trabalho de um
  notebook de Kaggle.
* **Diga as ressalvas em voz alta.** O caveat de exposição do analista, a ausência de
  custos reais e a ausência de evidência causal são pontos a favor, não contra.
* **Evite** dizer: "acurácia de 84 %", "o modelo descobriu que", "contrato mensal
  causa churn", "state of the art", "IA".
