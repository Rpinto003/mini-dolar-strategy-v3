# Mini Dólar Strategy v2

Estratégia de trading para mini dólar (WDO) combinando análises técnica, fundamental e machine learning.

## Estrutura do Projeto

```
src/
├── data/
│   ├── database/         # Dados brutos (candles.db)
│   ├── loaders/          # Carregadores de dados
│   └── collectors/       # Coletores (BCB, FRED)
├── analysis/
│   ├── technical/        # Análise técnica
│   └── fundamental/      # Análise fundamental
├── ml/
│   ├── models/           # Modelos base
│   ├── ensembles/        # Modelos ensemble
│   ├── features/         # Engenharia de features
│   └── evaluation/       # Avaliação dos modelos
├── reporting/
│   ├── summary/          # Sumários de performance
│   └── orders/          # Detalhamento das ordens
└── utils/              # Utilitários
```

## Componentes

### 1. Data
- Gerenciamento do banco de dados SQLite
- Coletores de dados econômicos (BCB, FRED)
- Loaders para diferentes timeframes

### 2. Análise
- Análise técnica com indicadores e padrões
- Análise fundamental com dados econômicos
- Integração de múltiplas fontes de dados

### 3. Machine Learning
- Modelos base (LSTM, Random Forest, XGBoost)
- Ensembles para maior robustez
- Feature engineering e seleção
- Avaliação e validação cruzada

### 4. Reporting
- Sumários de performance simplificados
- Relatórios detalhados de ordens
- Métricas principais de avaliação

## Instalação e Uso

Documentação em desenvolvimento.
