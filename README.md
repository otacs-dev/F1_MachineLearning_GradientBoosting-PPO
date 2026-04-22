# Projeto de IA para Previsão de Desempenho F1

Simulador de corridas e sistema de análise de dados para Fórmula 1 usando aprendizado de máquina e reinforcement learning.

## Estrutura do Projeto

| Pasta | Descrição |
|-------|-----------|
| **Circuit_List** | Dados e scripts de circuitos da F1 2025 |
| **Driver_List** | Dados e informações dos pilotos |
| **Race_List** | Dados de corridas e cache de informações |
| **Track_Results** | Resultados oficiais das corridas por rodada |
| **python_scripts_algorotimo_prep,trei,val** | Scripts de preparação e limpeza de dados do treinamento |
| **ml_ready** | Dataset pronto para modelagem (train/val/test) |
| **RL_F1_for_performance** | Simulador de corridas com Reinforcement Learning |

## Início Rápido

```bash
# Executar simulador interativo
cd RL_F1_for_performance
python test_env.py
```

## Fluxo de Dados

1. **Coleta** → Scripts extraem dados de circuitos, pilotos e resultados
2. **Processamento** → Limpeza, remoção de ruído e balanceamento
3. **ML Ready** → Dataset preparado em train/val/test
4. **Simulação** → Ambiente F1 com RL para análise de desempenho

## Requisitos

- Python 3.8+
- Bibliotecas: fastf1, pandas, numpy, matplotlib, tensorflow

## Documentação Detalhada

Consulte o README.md dentro de cada pasta para detalhes específicos.

---------------------------------------------------------------------

## Cronologia de Dados por Componente

### **Dados de Voltas e Performance (2024)**
- **RL_F1_for_performance/f1_data/lap_data.csv** → 2024 (26.608 voltas)
- **RL_F1_for_performance/teams_performance.py** → Temporada 2024

**Justificativa:** 
- Volume maior de dados de treino (~26k voltas)
- Melhor cobertura técnica por volta completa
- Baseline de performance consolidado

### **Resultados de Corridas (2025)**
- **Track_Results/Round_X_*_2025_results.csv** → Temporada 2025 completa
- **Circuit_List/F1_2025_Telemetry_Master.csv** → Telemetria 2025
- **Driver_List** → Roster de pilotos 2025
- **Race_List/Weather_2025_Master.csv** → Dados climáticos 2025
- **python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025*.csv** → Processamento 2025

**Justificativa:**
- Conformidade com calendário atual (abril/2026)
- Dados mais recentes de posições finais
- Atualizações de grid de pilotos 2025

### ** Nota Importante**
O projeto utiliza dados **híbridos mas sincronizados**:
- **Voltas de treino**: 2024 (maior volume, modelo técnico robusto)
- **Validação**: Contra resultados oficiais 2025 (posições reais)
