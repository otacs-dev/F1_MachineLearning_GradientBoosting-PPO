# Python Scripts - Preparação de Dataset

Scripts para limpeza, processamento e preparação de dados para treinamento.

## Scripts Principais

- **prepare_ml_dataset.py** - Orquestra todo o pipeline de preparação
- **clean_dataset.py** - Remove dados inválidos e duplicados
- **reduce_noise.py** - Filtra ruídos e outliers
- **balance_compounds.py** - Balanceia dados por tipo de pneu
- **add_context_flags.py** - Adiciona flags de contexto (pit stop, clima, etc)
- **edge_case_checks.py** - Valida casos extremos
- **validate_against_results.py** - Compara com resultados oficiais
- **integration_test.py** - Testa pipeline completo

## Datasets Gerados

- `Dataset_Treino_F1_2025.csv` - Dataset bruto
- `Dataset_Treino_F1_2025_CLEANED.csv` - Após limpeza
- `Dataset_Treino_F1_2025_DENOISED.csv` - Após remoção de ruído
- `Dataset_Treino_F1_2025_BALANCED.csv` - Dados balanceados
- `Dataset_Treino_F1_2025_FLAGS.csv` - Com flags de contexto

Executa pipeline completo de preparação dos dados.
