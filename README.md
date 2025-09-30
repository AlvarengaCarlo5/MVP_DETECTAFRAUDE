MVP - Detecção de Fraude em Transações (Classificação Supervisionada)
Tem por objetivo verificar se uma transação é fraudulenta (Class=1) ou legítima (Class=0). Devido a base de dados estar desbalanceada (com poucas fraudes) utilizei as métricas F1 (MACRO/PONDERADO) e AUC-PR (indicada para desbalamento), além de precisão, recall e matriz de confusão.

Abaixo segue roteiro para rodar localmente
I. Criar o ambiente "venv", em Python 3 é possível criar um ambiente virtual para rodar uma aplicação de forma isolada.
python -m venv .venv

II. Para ativarmos o ambiente virtual
.venv\Scripts\activate

III. Para instalar as dependências necessárias, como bibliotecas.
pip install -r requirements.txt

IV. Para rodar o treinamento
- Dividindo a base de dados em 80% treinamento e 20% testes, utilizando o algoritmo RandomForest (RF) e salvando o modelo e a config.
python train.py --data creditcard.csv --split random --test_size 0.20 --val_size 0.0 --threshold 0.5
- Caso queira deixar o modelo escolher o melhor threshold, por meio de avaliação, rode o cmando abaixo. Será escolhido o que maximiza o F1 na validação.
python train.py --data creditcard.csv --split random --test_size 0.20 --val_size 0.15

V. Caso deseje visualizar as métricas, rode o comando abaixo
type reports\metrics.json 
