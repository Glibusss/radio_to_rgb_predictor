# radio_to_rgb_predictor

Пайплайн `RGB -> radar grayscale` для случая, когда есть только одна пара: оптическое изображение и соответствующее ему радиолокационное изображение.

Сейчас в проекте две основные ветки решения:

- физический baseline, который классифицирует местность, назначает ЭПР/обратное рассеяние, строит полярное радарное изображение с ячейками `1 градус` по азимуту и `1.5 м` по дальности, а также учитывает затухание и радиотень;
- обучаемая модель `pix2pix`, которая использует оптические чанки, физически синтезированное дисплейное изображение и абсолютные координаты патча, чтобы предсказывать реальный радарный дисплей.

Лучший текущий результат даёт именно обучаемая display-модель:

- изображение: `outputs/pix2pix/pix2pix_coord_best_post.png`
- `masked_ncc = 0.6918`
- `masked_mae = 0.0672`

## Входные данные

Ожидаемые файлы в корне репозитория:

- `real_data.png` — оптическое RGB-изображение
- `radar_display.png` — целевое радарное изображение

Параметры радара, которые сейчас используются в коде:

- мощность передатчика: `50 W`
- высота антенны: `3 м`
- разрешение по дальности: `1.5 м`
- разрешение по азимуту: `1 градус`
- оптический GSD: `0.375 м/пиксель`
- рабочая частота: `9.4 GHz`

## Структура репозитория

Основные модули:

- `src/rgb2radio/terrain_resnet.py` — классификация местности и weak supervision
- `src/rgb2radio/rcs_library.py` — библиотека ЭПР / `sigma0` по классам
- `src/rgb2radio/physics_model.py` — физический синтез и формирование полярного дисплея
- `src/rgb2radio/chunking.py` — экспорт парных и непарных чанков
- `src/rgb2radio/pix2pix_display.py` — baseline `pix2pix`, привязанный к display-геометрии
- `src/rgb2radio/radar_reference.py` — очистка радарного дисплея и извлечение маски

CLI-утилиты:

- `tools/train_terrain_resnet.py`
- `tools/run_pipeline.py`
- `tools/make_chunks.py`
- `tools/train_pix2pix_display.py`
- `tools/extract_reference.py`

## Установка

Установка для CPU:

```powershell
python -m venv venv
./venv/Scripts/pip.exe install -r requirements-cpu.txt
```

В `requirements-cpu.txt` уже включены:

- `numpy`
- `opencv-python`
- `torch`

Если `torch` у вас уже установлен отдельно, можно использовать `requirements.txt` — там только минимальные зависимости без PyTorch.

## Быстрый старт

1. Обучить или обновить классификатор местности:

```powershell
./venv/Scripts/python.exe tools/train_terrain_resnet.py --epochs 10
```

2. Запустить физический пайплайн:

```powershell
./venv/Scripts/python.exe tools/run_pipeline.py
```

Основные артефакты физической ветки:

- `outputs/run/terrain_map_rgb.png`
- `outputs/run/projected_display_gray.png`
- `outputs/run/pure_physics_display_gray.png`
- `outputs/run/reference_clean_gray.png`
- `outputs/run/report.json`

3. Нарезать чанки для обучаемых моделей:

```powershell
./venv/Scripts/python.exe tools/make_chunks.py
```

4. Обучить baseline `pix2pix` для радарного дисплея:

```powershell
./venv/Scripts/python.exe tools/train_pix2pix_display.py --epochs 30 --batch-size 4
```

Основные артефакты обучаемой ветки:

- `outputs/pix2pix/pix2pix_display_gray.png`
- `outputs/pix2pix/pix2pix_coord_best_post.png`
- `outputs/pix2pix/pix2pix_report.json`
- `outputs/pix2pix/final_selection.json`

## Текущее состояние

Физический baseline с точечным display-рендером:

- строится по цепочке `классы -> ЭПР -> полярные resolution cells -> затухание/тень -> точечное отображение`
- текущие метрики из `outputs/run/report.json`:
  - `masked_ncc = 0.2821`
  - `masked_mae = 0.1356`

Обучаемая display-модель:

- `pix2pix` обучается на чанках `optical -> real radar display`
- входные каналы:
  - RGB-патч
  - патч физического радарного дисплея
  - нормализованная координата `x`
  - нормализованная координата `y`
- лучшая raw-модель:
  - `masked_ncc = 0.6914`
  - `masked_mae = 0.0539`
- лучший финальный кандидат после мягкого post-processing:
  - `masked_ncc = 0.6918`
  - `masked_mae = 0.0672`

## Важные замечания

- Проект был переписан под текущую структуру `src/` и `tools/` и не зависит от старой удалённой структуры.
- На текущий момент обучаемая модель заметно лучше повторяет целевое радарное изображение, чем строгая физическая ветка.
- Основное ограничение — данных очень мало: есть только одна оптическо-радарная пара.

## Что логично делать дальше

Самый сильный следующий эксперимент — residual cGAN:

- оставить текущее физическое изображение как основной структурный prior;
- обучать сеть предсказывать только поправку от физического дисплея к реальному радарному изображению;
- выбирать checkpoint по полной сборке сцены и метрике `masked_ncc`, а не только по patch-loss.
