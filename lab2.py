import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError

INPUT_DIR = os.path.join("data", "input")
OUTPUT_DIR = os.path.join("data", "output")
LOG_FILE = os.path.join("logs", "history.txt")
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.jpg")


def median_filter(img: Image.Image, size: int = 3) -> Image.Image:
    """Медианный фильтр (шумоподавление) — быстрая реализация через numpy."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    pad = size // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    
    # скользящее окно 
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(0, 1))
    
    # медиана 
    result = np.median(windows, axis=(-2, -1)).astype(np.uint8)
    return Image.fromarray(result)


def gamma_correction(img: Image.Image, gamma: float = 1.8) -> Image.Image:
    """Гамма-коррекция: out = 255 * (in/255) ** gamma."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(arr, gamma) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# Логирование 

def write_log(filename: str, filter_name: str, status: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] Файл: {filename} | Фильтр: {filter_name} | Статус: {status}\n"
    try:
      
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="UTF-8") as log:
            log.write(line)
    except OSError as e:
        print(f"Предупреждение: не удалось записать лог ({e})")




def main() -> None:
    filename = input("Введите имя файла для обработки (например, image1.jpg): ").strip()
    if not filename:
        print("Ошибка: имя файла не введено.")
        return

    path = os.path.join(INPUT_DIR, filename)
    filter_name = "Медианный фильтр + Гамма-коррекция"

    #  Проверка существования файла
    if not os.path.exists(path):
        print("Ошибка: Файл не найден")
        write_log(filename, filter_name, "Ошибка доступа")
        return

    #  Проверка прав на чтение
    if not os.access(path, os.R_OK):
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        write_log(filename, filter_name, "Ошибка доступа")
        return

    #  Открытие и обработка с перехватом исключений
    try:
        with Image.open(path) as img:
            img.load()  
            img = img.convert("RGB")

            # медианный фильтр
            filtered = median_filter(img, size=3)
            # Задание №2 — гамма-коррекция
            filtered = gamma_correction(filtered, gamma=1.8)

            # output
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filtered.save(RESULT_PATH, format="JPEG", quality=95)

        print(f"Готово: результат сохранён в {RESULT_PATH}")
        write_log(filename, filter_name, "Успешно")

    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        write_log(filename, filter_name, "Ошибка доступа")
    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        write_log(filename, filter_name, "Ошибка доступа")
    except (UnidentifiedImageError, OSError) as e:
        print(f"Ошибка: не удалось открыть изображение ({e})")
        write_log(filename, filter_name, "Ошибка доступа")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(0)
        