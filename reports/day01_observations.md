# Day 1 – EDA observations

Dataset: Financial PhraseBank  
Configuration: `sentences_75agree`  
Revision: `4f231b6c1b92db2a943966b01cc412cebb49f0f9`

## Data quality

The raw dataset contains 3453 rows.

There are:
- 0 rows with missing text
- 5 duplicated rows

After minimal structural cleaning, 3448 rows remain.

The dataset is therefore generally clean. Removing duplicates before future train/test splitting also reduces the risk of the same text appearing in both parts of the dataset.

## Class distribution

The cleaned dataset contains:

- neutral – 2141 samples, about 62.1%
- positive – 887 samples, about 25.7%
- negative – 420 samples, about 12.2%

The classes are noticeably imbalanced. Neutral is the majority class, while negative is the minority class.

For later model evaluation, macro F1 is therefore more informative than accuracy alone because it gives equal importance to all three classes.

A stratified train/test split should also be used so that the class proportions are preserved.

## Text length

Text length in characters:

- mean – 124.8
- median – 116
- minimum – 9
- maximum – 315

Word count:

- mean – 22.8
- median – 21
- minimum – 2
- maximum – 81

Most samples are relatively short financial sentences. This makes the dataset suitable for a classical TF-IDF baseline.

## Classes

The classification task contains three sentiment classes:

- negative
- neutral
- positive

The examples show that sentiment is based on the financial meaning of a sentence rather than general emotional language.

## Initial conclusion

The dataset is small and relatively clean, but it has a clear class imbalance.

The main issues to keep in mind for the next stages are:
- preserving class proportions during splitting
- evaluating the model with macro F1
- monitoring performance on the minority negative class
- keeping preprocessing consistent between training and inference
