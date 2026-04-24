# radio_to_rgb_predictor

Проект упрощен до одной задачи: классификация территории на изображении `real_data.png`.

Используемые классы:

1. Лес
2. Здание
3. Асфальтированная дорога
4. Грунтовая дорога
5. Вода
6. Тень от леса
7. Тень от здания
8. Кустарник
9. Автомобиль

## Основной запуск

```powershell
./venv/Scripts/python.exe tools/run_pipeline.py
```

Результат сохраняется в:

- `output/territories.png`

## Запуск с отладкой

```powershell
./venv/Scripts/python.exe tools/run_pipeline.py --debug
```

Дополнительно сохраняются промежуточные карты в:

- `output_debug/`

## Обучение классификатора местности

Если чекпойнта `outputs/models/terrain_resnet.pt` нет, пайплайн обучит его автоматически на `real_data.png`.

Отдельный запуск обучения:

```powershell
./venv/Scripts/python.exe tools/train_terrain_resnet.py
```

## Ключевые файлы

- `src/rgb2radio/common.py`
- `src/rgb2radio/terrain_resnet.py`
- `src/rgb2radio/territory_segmentation.py`
- `tools/run_pipeline.py`
- `tools/train_terrain_resnet.py`
