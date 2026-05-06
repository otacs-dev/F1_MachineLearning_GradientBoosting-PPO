Projeto de IA para Previsão de Desempenho F1 🏎️
Sistema avançado de simulação de corridas e análise preditiva de Fórmula 1, utilizando Deep Learning para previsão de tempos de volta e Reinforcement Learning (RL) para otimização de performance em tempo real.

###

📂 Arquitetura do Projeto
O projeto é dividido em quatro pilares principais, conforme a estrutura de diretórios:

1. ml_prepare_data (Preparação e Governança)
Esta pasta centraliza todo o pipeline de engenharia de dados.

Circuit_List: Metadados dos circuitos da temporada 2025 e telemetria master.
Race_List: Gerenciamento de clima (Weather_2025_Master.csv) e cache de dados via fastf1.
ml_ready_data: Armazena os datasets finais prontos para o consumo do modelo (train.csv, val.csv, test.csv).

Scripts de Limpeza:
clean_dataset.py: Remoção de outliers.
balance_compounds.py: Balanceamento de classes para diferentes tipos de pneus (Hard/Medium/Soft).

--------

2. training_session_algorithm (O Core da Inteligência)
Local onde ocorre o treinamento dos modelos neurais.

Execução: O modelo principal é treinado via train_v5_neural_laptime.py.
Outputs: A pasta training_output_v5 contém os modelos serializados (ex: .zip), logs detalhados da sessão e arquivos JSON de configuração.
Validação: O script validate_against_results.py confronta as previsões do modelo com os resultados reais dos 16 GPs de 2025 (da Austrália à Bélgica).

--------

3. RL_F1_for_performance (Simulador RL)
Ambiente de Reinforcement Learning para estratégia de corrida.

f1_env.py: O ambiente customizado (Gym/Gymnasium) que simula a dinâmica da pista.
f1_data: Contém o baseline técnico com lap_data.csv (26.608 voltas de 2024) e performances das equipes.
Componentes: Scripts dedicados para simular pitstop.py, race_events.py (Safety Cars/Bandeiras) e performance de pneus.

--------

4. others
Arquivos de suporte, rascunhos e documentação auxiliar.

🔄 Estratégia de Dados Híbrida (2024-2025)
Para garantir robustez e relevância, o projeto utiliza uma abordagem sincronizada:

Treinamento (Performance Base): Baseado em 2024.

Por que? Volume massivo de dados (~26k voltas) para ensinar ao modelo a física e o desgaste de pneus de forma estável.

Validação e Contexto (Grid Atual): Baseado em 2025.

Por que? Alinhamento com o grid de pilotos atual, telemetria 2025 e resultados reais de corrida (Track Results) para garantir que as previsões reflitam a ordem de forças atual da categoria.

######

Treinamento do Modelo Neural
Para iniciar uma nova sessão de treino apontando para a raiz do projeto:

(Bash)
python "training_session_algorithm/train_v5_neural_laptime.py" --data-root /caminho/para/PI_IV
Teste do Ambiente de Simulação (RL)
Para interagir com o simulador e observar o comportamento do agente:

(Bash)
cd RL_F1_for_performance
python test_env.py

#####

🛠️ Requisitos Técnicos
Python 3.8+

Core: tensorflow, pandas, numpy, fastf1
Visualização: matplotlib, seaborn
RL: gymnasium ou stable-baselines3

consultar "venv" caso tenha dúvidas de dependência


