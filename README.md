# Hotel Booking Cancellation Prediction

**Predicting which hotel bookings will be cancelled, and using a genetic algorithm to choose which features matter.**

A machine-learning project on 119k real hotel bookings. It covers data cleaning, exploratory analysis, three classifiers, and **genetic-algorithm feature selection**, which cut the feature set by 25–52% while *improving* accuracy for every model.

![Python](https://img.shields.io/badge/Python-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![GA](https://img.shields.io/badge/feature%20selection-genetic%20algorithm-8E44AD)

---

## Approach

1. **Cleaning**: drop leaky and sparse columns (`reservation_status`, `company`, `agent`, …), impute missing values (median / mode), treat outliers, remove duplicates
2. **Feature engineering**: frequency encoding for hotel and country, month-to-number mapping, total stay length, one-hot categoricals → 44 features
3. **EDA**: cancellation rate by month, week, lead time, ADR, deposit type, market segment, customer type, special requests and previous cancellations
4. **Models**: Decision Tree, K-Nearest Neighbours, Multi-Layer Perceptron
5. **GA feature selection**: an evolutionary search (`sklearn-genetic-opt`) over feature subsets for each model, then a comparison of all features vs the GA-selected subset

## Results

Test-set results, all 44 features → GA-selected subset:

| Model | Features kept | Accuracy | F1 |
|---|---|---|---|
| Decision Tree | 27 (−39%) | 0.806 → **0.825** | 0.743 → **0.769** |
| KNN | 21 (−52%) | 0.780 → **0.836** | 0.699 → **0.780** |
| MLP | 33 (−25%) | 0.841 → **0.845** | 0.773 → **0.782** |

GA selection improved every model while using fewer features. The biggest gain was for KNN (+5.5 accuracy points, +8.1 F1 points), which is the model most hurt by irrelevant dimensions. The MLP is the best model overall.

## Run it

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn sklearn-genetic-opt
jupyter notebook hotel_cancellation.ipynb
```

## Files

| File | What |
|---|---|
| `hotel_cancellation.ipynb` | Full analysis and models |
| `hotel_bookings.csv` | Dataset: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Antonio, Almeida & Nunes, 2019) |
| `Report.pdf` | Written report |
