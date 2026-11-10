# Neural Networks Labs

Збірник лабораторних робіт з **глибокого навчання** на Keras (бекенд PyTorch).

Від повнозв’язних мереж і підбору гіперпараметрів — до CNN, autoencoders, LSTM/GRU та прогнозування часових рядів.

---

## Огляд модулів

| Модуль | Теми | Ноутбуки |
|--------|------|----------|
| [Lab 1](lab%201/) | MLP: регресія, класифікація, time series | 3 |
| [Lab 2](lab%202/) | CNN на MNIST, перебір конфігурацій | 1 |
| [Lab 3](lab%203/) | CIFAR-10: CNN + ConvNeXt V2 | 2 |
| [Lab 4](lab%204/) | Autoencoders (dense + conv) на Fashion-MNIST | 1 |
| [Lab 5](lab%205/) | NLP: LSTM / BiLSTM+Conv1D | 2 |
| [Lab 6](lab%206/) | GRU для bitcoin time series | 1 |

---

## Структура репозиторію

```
neural-networks-labs/
├── lab 1/          # MLP
├── lab 2/          # CNN (MNIST)
├── lab 3/          # CIFAR-10 (+ images/)
├── lab 4/          # Autoencoders
├── lab 5/          # RNN / LSTM
├── lab 6/          # GRU time series
├── requirements.txt
└── README.md
```

---

## Lab 1 — Dense MLP

Перебір архітектур, активацій і оптимізаторів; окремо — послідовна модель для bitcoin.

| Ноутбук | Зміст |
|---------|--------|
| `lab1_task1.ipynb` | `hourly_wages_data.csv` — регресія, 5 архітектур × 5 активацій × 7 оптимізаторів |
| `lab1_task2.ipynb` | `diabetes_data.csv` — бінарна класифікація (`softmax` + `categorical_crossentropy`) |
| `lab1_task3.ipynb` | Bitcoin hourly indicators — Flatten + Dense MLP, MSE |

---

## Lab 2 — CNN (MNIST)

| Ноутбук | Зміст |
|---------|--------|
| `lab2.ipynb` | Класифікація цифр: Conv2D-блоки, перебір ядер / глибини / оптимізаторів, таблиця `results_df` |

> `train.csv` / `test.csv` локально (у `.gitignore`).

---

## Lab 3 — CIFAR-10

| Ноутбук | Зміст |
|---------|--------|
| `lab_3_task1.ipynb` | Класична CNN + augmentation (`ImageDataGenerator`) |
| `lab_3_task2.ipynb` | Micro ConvNeXt V2 (DepthwiseConv, GRN, LayerNorm, GELU) + інференс на `images/` |

---

## Lab 4 — Autoencoders

| Ноутбук | Зміст |
|---------|--------|
| `lab4.ipynb` | Fashion-MNIST: dense AE (latent 32) і convolutional AE; реконструкція / denoising |

---

## Lab 5 — Recurrent NLP

| Ноутбук | Зміст |
|---------|--------|
| `lab5_task1.ipynb` | SMS Spam → Embedding + LSTM |
| `lab5_task2.ipynb` | IMDB → Embedding + Conv1D + Bidirectional LSTM |

---

## Lab 6 — GRU Time Series

| Ноутбук | Зміст |
|---------|--------|
| `lab_6.ipynb` | Bitcoin hourly indicators → Masking + stacked GRU, MSE, криві навчання |

> CSV з індикаторами локально (у `.gitignore`), як і для lab 1 task 3.

---

## Швидкий старт

```bash
git clone https://github.com/YaTut1901/neural-networks-labs.git
cd neural-networks-labs

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

У ноутбуках перед `import keras` виставляється:

```python
os.environ["KERAS_BACKEND"] = "torch"
```

---

## Залежності

- Python 3.10+ (рекомендовано)
- Keras 3 + PyTorch (CPU wheels у `requirements.txt`)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, scikit-image
- Jupyter / ipykernel

Повний список — у [`requirements.txt`](requirements.txt).
