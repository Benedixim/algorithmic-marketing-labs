# Лабораторные по алгоритмическому маркетингу

Python-код для лабораторных №1, 2, 4–10. Лабораторной №3 в предоставленном задании нет.

| Работа | Скрипт | Что выполняет |
| --- | --- | --- |
| 1 | [lab01.py](code/lab01.py) | Анализ покупок и кластеризация клиентов |
| 2 | [lab02.py](code/lab02.py) | Рекомендации товаров и сравнение с популярными товарами |
| 4 | [lab04.py](code/lab04.py) | Модель продаж и распределение рекламного бюджета |
| 5 | [lab05.py](code/lab05.py) | Классификация тональности отзывов |
| 6 | [lab06.py](code/lab06.py) | Анализ связи цены и объёма продаж |
| 7 | [lab07.py](code/lab07.py) | Ранжирование товаров по повторным покупкам |
| 8 | [lab08.py](code/lab08.py) | RFM-сегментация клиентов |
| 9 | [lab09.py](code/lab09.py) | Подготовка сегментов и генерация рекламных текстов |
| 10 | [lab10.py](code/lab10.py) | Прогнозирование оттока клиентов |

Графики строятся в [make_charts.py](code/make_charts.py) с помощью Matplotlib. Общие функции находятся в [common.py](code/common.py).

## Запуск

Нужны Python 3.11+ и зависимости из `requirements.txt`. Команды выполняются из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/download_data.py
python code/run_all.py
```

В Windows активируйте окружение командой `.venv\Scripts\activate`.
Можно запускать работы отдельно, например `python code/lab01.py`.
JSON и CSV с результатами сохраняются в `results/`. При первом чтении Online Retail создаётся локальный кэш `data/retail.pkl`.

Датасеты не включены в репозиторий. Скрипт загрузки использует открытые источники из [sources.json](sources.json) и проверяет SHA-256 загруженных файлов.

## Лабораторная №9 и графики

Обычный запуск `run_all.py` только подготавливает входные данные для модели. Для генерации нужны Node.js 22+ и npm:

```bash
npm install
python code/lab09.py
python code/make_charts.py
```

Скрипт [generate.mjs](code/generate.mjs) запускает Qwen2.5-1.5B-Instruct через Transformers.js на CPU. При первом запуске скачиваются публичные веса ONNX q4, около 1,7 ГБ, в `model-cache/`. Платный API и ключи не нужны. Генерация может занимать значительное время.

`make_charts.py` читает рассчитанные результаты, включая результат генерации ЛР 9, и сохраняет PNG в `figures/`.

## Источники и смысл результатов

- Задания курса: [СЭО БГУИР, Пархименко](https://lms.bsuir.by/course/view.php?id=3233).
- Покупки: [UCI Online Retail](https://doi.org/10.24432/C5BW33), CC BY 4.0. Период 01.12.2010–09.12.2011.
- Реклама: [Advertising, ISLR](https://www.statlearning.com/resources-second-edition), учебный пример из 200 рынков.
- Отзывы: [UCI Sentiment Labelled Sentences](https://doi.org/10.24432/C57604), CC BY 4.0, файл Amazon Cells.
- Отток: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), учебные данные вымышленной компании.
- Модель: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache 2.0, [ONNX-версия](https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct).

Изменение продаж после оптимизации бюджета представляет модельный сценарий. Наблюдаемая связь цены и объёма не доказывает причинный эффект скидки. Рейтинг в ЛР 7 основан на повторных покупках за 60 дней. RFM-группы заданы эвристически. Отток проверяется на случайной отложенной выборке учебных данных. Формальные проверки рекламного текста не измеряют эффективность рекламы.
