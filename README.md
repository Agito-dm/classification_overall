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
|---|---:|
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
- baseline pipeline `TF-IDF + Logistic Regression`
- stratified train/test split 80/20
- расчёт Accuracy, Macro F1 и Weighted F1
- classification report с метриками по каждому классу
- сохранение baseline-метрик и модели
- автоматические тесты baseline pipeline и метрик

Метки классов:

| Класс | Label |
|---|---:|
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

Обучение baseline-модели:

```powershell
python -m finnews_sentiment.models.train_model
```

## Результаты

Отчёты и графики сохраняются в `reports/`:

```text
reports/
├── day01_statistics.txt
├── day01_observations.md
├── day02_preprocessing_report.txt
├── day02_examples.md
├── day03_baseline_metrics.json
├── day03_classification_report.txt
├── day04_model_comparison.csv
├── day04_model_comparison.json
├── day04_best_classification_report.txt
└── figures/
    ├── sentiment_distribution.png
    ├── text_length_distribution.png
    └── day04_macro_f1_comparison.png
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
├── models/
├── notebooks/
├── reports/
├── src/
│   └── finnews_sentiment/
│       ├── data/
│       ├── features/
│       ├── models/
│       │   ├── compare_models.py
│       │   └── train_model.py
│       └── visualization/
├── tests/
│   ├── test_compare_models.py
│   ├── test_preprocess.py
│   └── test_train_model.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Baseline модель

В качестве первой точки отсчёта используется классический pipeline:

```text
text_clean
→ TF-IDF
→ Logistic Regression
```

Данные разделяются на train и test в пропорции 80/20 с сохранением распределения классов (`stratify`).

Параметры baseline хранятся в:

```text
configs/train_baseline.json
```

Запуск обучения:

```powershell
python -m finnews_sentiment.models.train_model
```

Текущий baseline:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.8348 |
| Macro F1 | 0.7530 |
| Weighted F1 | 0.8216 |

F1 по классам:

| Класс | F1 |
|---|---:|
| negative | 0.6290 |
| neutral | 0.8957 |
| positive | 0.7342 |

Baseline заметно лучше распознаёт преобладающий класс `neutral`, в то время как recall класса `negative` составляет 0.4643. Поэтому основной метрикой дальнейшего сравнения моделей используется Macro F1.

Результаты сохраняются в:

```text
reports/day03_baseline_metrics.json
reports/day03_classification_report.txt
```

Локально также создаётся baseline-модель:

```text
models/baseline_logreg.joblib
```

Сравнение вариантов модели:

```powershell
python -m finnews_sentiment.models.compare_models
```


```markdown
- сохранение baseline-метрик и модели
- автоматические тесты baseline pipeline и метрик
- сравнение unigram/bigram TF-IDF, balanced Logistic Regression и LinearSVC
- автоматический выбор лучшей модели по Macro F1
- сохранение сравнительных метрик, classification report и графика
```

## Улучшение модели

Проверены три изменения baseline:

| Вариант | Macro F1 |
|---|---:|
| Baseline Logistic Regression | 0.7530 |
| TF-IDF bigrams + Logistic Regression | 0.7464 |
| Balanced Logistic Regression | 0.8015 |
| LinearSVC | 0.8113 |

Лучший результат показал `LinearSVC`.

Macro F1 вырос с `0.7530` до `0.8113`:

- абсолютный прирост – `+0.0583`
- относительный прирост – `+7.74%`

Использование bigrams отдельно улучшения не дало, а учёт дисбаланса классов заметно повысил качество Logistic Regression.
