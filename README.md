# Financial News Sentiment Classification

Проект по классификации тональности финансовых новостей.

Цель – построить воспроизводимый ML pipeline от загрузки и анализа данных до обучения модели, оценки качества и inference.

## Данные

Используется датасет Financial PhraseBank:

- источник – `takala/financial_phrasebank`
- конфигурация – `sentences_75agree`
- классы – `negative`, `neutral`, `positive`
- количество исходных записей – 3453

Датасет скачивается автоматически и не хранится в Git.

После первичной очистки:

- удалено 5 дубликатов
- пропущенных текстов не обнаружено
- осталось 3448 записей

Распределение классов:

| Класс | Количество |
| neutral | 2141 |
| positive | 887 |
| negative | 420 |

В данных присутствует заметный дисбаланс классов, поэтому при дальнейшем обучении основной метрикой будет Macro F1.

## Реализовано

На текущем этапе проекта:

- воспроизводимая загрузка Financial PhraseBank
- проверка структуры и качества данных
- анализ распределения классов
- анализ длины текстов
- удаление дубликатов
- базовая очистка и нормализация текста
- сохранение финансово значимых чисел, процентов и валютных обозначений
- преобразование классов в числовые метки
- проверка пустых текстов, новых дубликатов и конфликтующих меток
- простые диагностические признаки текста
- автоматические тесты preprocessing

Метки классов:

| Класс | Label |
| negative | 0 |
| neutral | 1 |
| positive | 2 |

После preprocessing осталось 3448 записей. Новых дубликатов, пустых текстов и конфликтующих меток не обнаружено.

## Запуск

Рекомендуемая версия Python – 3.10.

Создание окружения:

```powershell
conda create -n classification_overall python=3.10
conda activate classification_overall
python -m pip install -r requirements.txt
```

Загрузка датасета:

```powershell
python -m finnews_sentiment.data.download_data
```

Запуск анализа данных:

```powershell
python -m finnews_sentiment.visualization.eda
```

Запуск preprocessing:

```powershell
python -m finnews_sentiment.features.preprocess
```

Запуск тестов:

```powershell
python -m unittest discover -s tests -v
```

## Результаты

Отчёты и графики сохраняются в `reports/`:

```text
reports/
├── day01_statistics.txt
├── day01_observations.md
├── day02_preprocessing_report.txt
├── day02_examples.md
└── figures/
    ├── sentiment_distribution.png
    └── text_length_distribution.png
```

Обработанный датасет создаётся локально:

```text
data/processed/financial_phrasebank_75agree_processed.csv
```

Raw и processed данные исключены из Git и воспроизводятся командами проекта.

## Структура проекта

```text
classification_overall/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
├── src/
│   └── finnews_sentiment/
│       ├── data/
│       ├── features/
│       ├── models/
│       └── visualization/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```