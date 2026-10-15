# Financial News Sentiment Classification

ML проект для определения тональности финансовых новостей на английском языке: **negative**, **neutral** или **positive**.

Проект охватывает весь цикл классификации текста: загрузку данных, очистку, обучение baseline, сравнение моделей, анализ ошибок и inference новых сообщений. Для финального решения выбрана связка **TF-IDF (unigrams) + LinearSVC**.

## Быстрый старт

Команды ниже выполняются **из корня репозитория**. Для загрузки датасета требуется интернет. Рекомендуемая версия — **Python 3.10**.

```powershell
# Если проект ещё не клонирован:
git clone https://github.com/Agito-dm/classification_overall.git
cd classification_overall

# Окружение и зависимости:
conda create -n classification_overall python=3.10
conda activate classification_overall
python -m pip install -r requirements.txt
```

Подготовить данные, обучить финальную модель и проверить предсказание:

```powershell
python -m finnews_sentiment.data.download_data
python -m finnews_sentiment.features.preprocess
python -m finnews_sentiment.models.train_best_model
python -m finnews_sentiment.models.predict --text "The company reported strong revenue growth."
```

Запустить единый demo с семью примерами:

```powershell
python -m finnews_sentiment.demo
```

Demo выводит предсказания в терминал и сохраняет их в `reports/demo_predictions.json`. Для inference используется сохранённая модель `models/best_model.joblib`: **повторное обучение при каждом предсказании не требуется**.

## Inference: вход и выход

CLI принимает **один или несколько исходных английских текстов**. Для нескольких сообщений повторите `--text`:

```powershell
python -m finnews_sentiment.models.predict --text "Revenue increased by 20 percent." --text "Sales declined last quarter."
```

Результат для каждого текста содержит:

- `text` — исходное предложение;
- `sentiment` — `negative`, `neutral` или `positive`;
- `label` — соответствующая числовая метка (`0`, `1`, `2`).

Очистка текста (`clean_text`) выполняется автоматически, точно так же, как при подготовке обучающих данных. Пустые строки после очистки отклоняются с понятной ошибкой. Для использования из Python доступны функции `load_model()` и `predict_sentiment()` в `finnews_sentiment.models.predict`.

Сохранённый `joblib`-файл следует загружать **только из доверенного источника**.

## Демонстрация

Demo использует финальную модель, обученную на всех 3448 обработанных примерах. Ниже — реальные предсказания, полученные при запуске:

| Сценарий | Предсказание |
|---|---|
| Revenue growth | `positive` |
| Corporate meeting | `neutral` |
| Falling profit | `negative` |
| Lower operating costs | `negative` |
| Mixed financial signals | `negative` |
| Product announcement | `neutral` |
| Uncertain outlook | `neutral` |

Последние случаи показывают не только возможности, но и ограничения модели. Например, снижение операционных затрат было отмечено как `negative`, а предупреждение о возможном ослаблении спроса — как `neutral`. Это **иллюстрации поведения**, а не новая оценка Accuracy на семи примерах: независимая эталонная разметка для demo не проводилась.

Тексты сценариев находятся в `src/finnews_sentiment/demo.py`, результаты — в `reports/demo_predictions.json`. Предсказанные классы не зашиты в код demo.

## Данные и подготовка

Используется [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), конфигурация `sentences_75agree`.

- Исходных предложений: **3453**.
- Удалено точных дубликатов: **5**.
- После очистки: **3448**.
- Отсутствующих текстов, новых дубликатов после preprocessing и конфликтующих меток не обнаружено.

| Класс | Метка | Число примеров |
|---|---:|---:|
| negative | 0 | 420 |
| neutral | 1 | 2141 |
| positive | 2 | 887 |

Preprocessing включает приведение к нижнему регистру, удаление HTML-тегов и нормализацию пробелов. Числа, валютные обозначения и проценты сохраняются, так как они важны для финансовых сообщений. Датасет с подготовленными текстами создаётся в `data/processed/financial_phrasebank_75agree_processed.csv`.

Исходные и обработанные CSV не хранятся в Git: они воспроизводятся командами из раздела «Быстрый старт».

## Модели и результаты

Для сравнения использовался один и тот же **stratified train/test split 80/20** (`random_state=42`): 2758 примеров для обучения и 690 для оценки. Основная метрика — **Macro F1**, поскольку классы несбалансированы.

| Конфигурация | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF unigram + Logistic Regression (baseline) | 0.8348 | 0.7530 |
| TF-IDF unigram/bigram + Logistic Regression | 0.8275 | 0.7464 |
| TF-IDF unigram + balanced Logistic Regression | 0.8449 | 0.8015 |
| **TF-IDF unigram + LinearSVC** | **0.8609** | **0.8113** |

По сравнению с baseline, Macro F1 лучшей конфигурации увеличился на **0.0583** (5.83 процентного пункта, или **7.74% относительно baseline**). Добавление bigrams само по себе улучшения не дало; изменение весов классов помогло Logistic Regression.

**Важно:** `0.8113` — результат holdout-оценки до финального переобучения. После выбора конфигурации `train_best_model.py` обучает её на **всех 3448 примерах** и сохраняет `models/best_model.joblib` для inference. Эта финальная модель **не имеет отдельной независимой тестовой метрики**.

## Анализ ошибок

Для выбранного `LinearSVC` на holdout-выборке получено **594 верных предсказания и 96 ошибок** из 690 примеров.

Основные направления ошибок:

| Истинный класс → предсказанный | Число ошибок |
|---|---:|
| positive → neutral | 42 |
| negative → neutral | 19 |
| neutral → positive | 18 |
| negative → positive | 13 |

**61 из 96 ошибок** связаны с отнесением положительных или отрицательных сообщений к `neutral`. Для `negative` recall составляет **0.6190**. Модель также может неверно интерпретировать экономический смысл снижения показателя: уменьшение затрат не всегда означает плохую новость.

Подробности — в `reports/day05_error_analysis.md`, `reports/day05_errors.csv` и `reports/figures/day05_confusion_matrix.png`.

## Воспроизведение экспериментов и тесты

После загрузки и preprocessing можно отдельно повторить анализ, обучение baseline, сравнение и анализ ошибок:

```powershell
python -m finnews_sentiment.visualization.eda
python -m finnews_sentiment.models.train_model
python -m finnews_sentiment.models.compare_models
python -m finnews_sentiment.models.error_analysis
```

Автоматические тесты и проверка установленных зависимостей:

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

В `reports/` сохраняются метрики, classification reports, таблицы ошибок и изображения. Основные создаваемые артефакты:

| Путь | Назначение |
|---|---|
| `models/best_model.joblib` | Финальный обученный Pipeline: TF-IDF + LinearSVC |
| `reports/day04_model_comparison.csv` | Сравнение четырёх конфигураций |
| `reports/day05_error_analysis.md` | Причины и примеры ошибок |
| `reports/day05_errors.csv` | Ошибочные примеры из holdout-выборки |
| `reports/demo_predictions.json` | Реальные результаты demo |

## Структура проекта

```text
classification_overall/
├── configs/                        # Параметры обучения
├── data/                           # raw/ и processed/ (генерируются локально)
├── docs/                           # Дополнительная документация и checklist
├── models/                         # Сохранённые модели (генерируются локально)
├── notebooks/
├── reports/                        # Метрики, анализ и графики
├── src/finnews_sentiment/
│   ├── data/                        # Загрузка датасета
│   ├── features/                    # Очистка и preprocessing
│   ├── models/
│   │   ├── best_model.py            # Конфигурация выбранной модели
│   │   ├── compare_models.py        # Сравнение экспериментов
│   │   ├── error_analysis.py        # Confusion matrix и ошибки
│   │   ├── predict.py               # Inference API и CLI
│   │   ├── train_best_model.py      # Обучение и сохранение финальной модели
│   │   └── train_model.py           # Baseline
│   ├── visualization/               # EDA
│   └── demo.py                      # Единый исполняемый demo
├── tests/                           # Автоматические unit/integration tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Ограничения

- **Область применения:** модель обучена на небольшом англоязычном корпусе финансовых предложений. Качество на других языках, источниках и типах текстов отдельно не измерялось.
- **Дисбаланс классов:** `neutral` заметно преобладает, а положительные и отрицательные высказывания часто ошибочно относятся к нейтральным.
- **Контекст:** unigram TF-IDF учитывает отдельные слова, но не моделирует полноценно отношения между ними, отрицания, финансовый смысл изменений и смешанные сигналы.
- **Оценка:** одна и та же holdout-выборка использовалась для сравнения кандидатов и отчётной метрики; отдельного нетронутого финального test set не было. Результат может быть оптимистичнее независимой оценки.
- **Уверенность:** `LinearSVC` не выдаёт калиброванные вероятности классов по умолчанию; предсказанная метка не означает гарантированной уверенности или правильности.
- **Назначение:** предсказание тональности новости не равно прогнозу цены акций или инвестиционной рекомендации.

## Воспроизводимость

Репозиторий содержит код для повторной загрузки данных, preprocessing, обучения модели и запуска demo. Данные и бинарные модели **не версионируются в Git** и должны генерироваться командами выше. Рабочая директория для приведённых команд — корень клонированного проекта.

Для независимой проверки следует повторить **«Быстрый старт» и тесты в новом клоне и новом Python-окружении**, не используя данные и модели из исходной рабочей папки. Контрольный список находится в `docs/reproducibility_checklist.md`.
