# RL_F1_for_performance - Simulador de Corridas com RL

Simulador interativo de Fórmula 1 com Reinforcement Learning para análise de desempenho.

## Arquivos Principais

- **test_env.py** - Interface interativa do simulador (execute isto!)
- **f1_env.py** - Ambiente de simulação (gym/gymnasium compatible)
- **lapdata.py** - Processamento de dados de volta
- **pitstop.py** - Estratégia de pit stop
- **race_events.py** - Eventos de corrida (acidentes, neutralização, etc)
- **teams_performance.py** - Análise de desempenho de equipes
- **eda_inicial.py** - Análise exploratória inicial

Interface de terminal interativa que pede:
1. Escolha do circuito
2. Escolha da equipe
3. Escolha do piloto
4. Posição na grid (1-20 ou AUTO)

- Histórico de posições
- Gaps entre pilotos
- Locais de pit stop
- Tipos de pneu utilizados

## Características

- Simulação realista de corridas F1
- Suporte a estratégias de pit stop
- Visualização dinâmica de posições
- Performance metrics em tempo real
