# Análise de Sentimento em Tweets

O aumento uso das redes sociais e a disponibilização de quantidades massivas de textos online fez surgir o interesse em se analisar e compreender o conteúdo disponível nos mais diversos meios. Uma das frentes de pesquisa que se encarregar de tal tarefa é a Mineração de Argumentos, cujo objetivo é identificar, extrair e compreender a estrutura argumentativa de textos online (Sousa et al. 2021). Como toda grande tarefa, a mineração de argumentos é dividida em conjuntos menores de trabalhos que podem ser realizados individualmente. Um deles — talvez o mais popular — é a análise de sentimento, que consiste em identificar se um dado documento ou pedaço de texto carrega uma conotação positiva, negativa ou neutra.

Partindo das definições acima, o objetivo do presente trabalho é executar a análise de sentimentos contidos em [Tweets coletados entre 01/08/2018 e 20/10/2018](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis).


## Estrutura do Repositório

------------

    ├── LICENSE
    ├── README.md 
    ├── data
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── processed        <- The final, canonical data sets for modeling.
    │   └── raw              <- The original, immutable data dump.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks            <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                           the creator's initials, and a short `-` delimited description, e.g.
    │                           `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment, e.g.
    │                           generated with `pip freeze > requirements.txt`
    │
    ├── setup.py             <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                  <- Source code for use in this project.
    │   │
    │   ├─ build_dataset.py  <- Scripts to download or generate data
    │   ├─ build_features.py <- Scripts to turn raw data into features for modeling
    │   ├─ train.py          <- Scripts to train models
    │   ├─ predict.py        <- Scripts to use trained models to make predictions
    │   └─ visualize.py      <- Script to create exploratory and results oriented visualizations
    │ 

---

## Referências

> Sousa, João Pedro da Silva, Rodrigo Costa Uchoa do Nascimento, Renata Mendes de Araujo, e Orlando Bisacchi Coelho. 2021. “Não se perca no debate! Mineração de Argumentação em Redes Sociais”. P. 139–50 em Anais do Brazilian Workshop on Social Network Analysis and Mining (BraSNAM). SBC.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
