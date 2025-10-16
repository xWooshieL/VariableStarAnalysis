# VariableStarAnalysis (Python)

> Python-версия проекта анализа переменных звёзд, портированная с оригинального Java-репозитория [sevenstarknight/VariableStarAnalysis](https://github.com/sevenstarknight/VariableStarAnalysis)

## Описание

Этот проект предназначен для анализа переменных звёзд с использованием современных методов машинного обучения и статистического анализа. Реализация включает:
- Анализ производных полей (Derivatives Fields)
- Модели на основе марковских цепей (Semi-Supervised Markov Models)
- Извлечение признаков из временных рядов фотометрических данных
- Утилиты для работы с астрономическими данными

## Структура проекта

```
VariableStarAnalysis_py/
├── analysis/
│   ├── markov/                  # Марковские цепи, semi-supervised модели
│   ├── df/                      # Derivatives fields анализ
│   └── feature_engineering/     # Фиче-инжиниринг, выделение признаков
├── lib/                         # Общие утилиты (matrix_utils.py и т.п.)
├── data/                        # Датасеты/результаты (необязательно)
├── tests/                       # Тесты (pytest)
├── requirements.txt             # Зависимости Python
├── README.md                    # Документация проекта
├── .gitignore                   # Игнорируемые файлы
└── LICENSE                      # Лицензия проекта (GPLv3)
```

### Описание модулей:
- **analysis/**: основные модули анализа переменных звёзд
  - `markov/` — реализация марковских цепей и semi-supervised моделей для классификации
  - `df/` — анализ производных полей (derivatives fields) кривых блеска
  - `feature_engineering/` — извлечение признаков из фотометрических временных рядов
- **lib/**: общие утилиты и вспомогательные функции (матричные операции, конвертеры и т.д.)
- **data/**: папка для хранения датасетов и результатов анализа
- **tests/**: unit-тесты для всех модулей (pytest)

## Установка

### Требования
- Python 3.8+
- pip или conda

### Установка зависимостей

```bash
# Клонируйте репозиторий
git clone https://github.com/YOUR_USERNAME/VariableStarAnalysis_py.git
cd VariableStarAnalysis_py

# Создайте виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
venv\Scripts\activate  # для Windows

# Установите зависимости
pip install -r requirements.txt
```

## Использование

### Базовый пример анализа

```python
from analysis.feature_engineering import extract_features
from analysis.markov import MarkovClassifier
import pandas as pd

# Загрузка данных
data = pd.read_csv('data/lightcurves.csv')

# Извлечение признаков
features = extract_features(data)

# Классификация с помощью марковских моделей
classifier = MarkovClassifier()
classifier.fit(features)
predictions = classifier.predict(features)
```

### Запуск анализа из командной строки

```bash
# Пример запуска анализа derivatives fields
python -m analysis.df --input data/input.csv --output results/

# Пример обучения марковской модели
python -m analysis.markov --train data/train.csv --model models/markov_model.pkl
```

## Основные зависимости

- **numpy** — матричные вычисления и линейная алгебра
- **pandas** — обработка табличных данных
- **scipy** — научные вычисления и статистика
- **scikit-learn** — машинное обучение
- **astropy** — работа с астрономическими данными
- **matplotlib/seaborn** — визуализация результатов

Полный список зависимостей см. в `requirements.txt`.

## Разработка и тестирование

### Запуск тестов

```bash
# Запуск всех тестов
pytest tests/

# Запуск с покрытием кода
pytest --cov=analysis --cov=lib tests/
```

### Структура тестов

```
tests/
├── test_markov.py
├── test_df_analysis.py
├── test_feature_engineering.py
└── test_utils.py
```

## Contributing

Мы приветствуем вклад в развитие проекта! Чтобы внести свой вклад:

1. Сделайте fork репозитория
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## Лицензия

Этот проект распространяется под лицензией **GNU General Public License v3.0 (GPLv3)**.

Это означает, что вы можете:
- Свободно использовать код для любых целей
- Модифицировать и распространять изменённые версии
- Использовать в коммерческих проектах

При условии, что производные работы также будут распространяться под GPLv3 и с открытым исходным кодом.

Подробнее см. [LICENSE](LICENSE).

## Авторы

- **Оригинальный проект (Java)**: [sevenstarknight](https://github.com/sevenstarknight/VariableStarAnalysis)
- **Python-версия**: [Никита Новицкий](https://github.com/xWooshieL), [Никита Борисов](https://github.com/nizier193), [Глеб Дудоров](https://github.com/Udnek)

## Благодарности

- Оригинальному автору Java-версии проекта за разработку алгоритмов и методологии
- Центральному Университету за поддержку исследований
- Научной студии по переменным звёздам

## Контакты

Если у вас есть вопросы или предложения:
- Создайте Issue в репозитории
- Напишите одному из нас на email: n.novitskiy@edu.centraluniversity.ru, n.borisov@edu.centraluniversity.ru, g.dudorov@edu.centraluniversity.ru

## Roadmap

- [ ] Полный порт всех модулей с Java на Python
- [ ] Добавление новых методов классификации переменных звёзд
- [ ] Интеграция с современными ML-библиотеками (PyTorch, TensorFlow)
- [ ] Веб-интерфейс для визуализации результатов
- [ ] Документация API (Sphinx)
- [ ] CI/CD pipeline (GitHub Actions)

## Ссылки

- [Оригинальный Java-репозиторий](https://github.com/sevenstarknight/VariableStarAnalysis)
- [Документация по переменным звёздам](https://www.aavso.org/vstar-overview)
- [Astropy Documentation](https://docs.astropy.org/)
