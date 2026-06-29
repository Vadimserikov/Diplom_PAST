#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B тестирование: нейросеть vs эвристика
Для дипломной работы
"""

import json
import csv
import os
from transformers import pipeline

# ========== ЗАГРУЗКА МОДЕЛЕЙ ==========
print("🔄 Загрузка нейросетевых моделей...")

sentiment_pipeline = pipeline("sentiment-analysis", model="cointegrated/rubert-tiny-sentiment-balanced")
toxicity_pipeline = pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity")

_hidden_model = None
model_path = "./model_hidden_abuse"
if os.path.exists(model_path) and os.listdir(model_path):
    try:
        _hidden_model = pipeline("text-classification", model=model_path, tokenizer=model_path)
        print("✅ Дообученная модель загружена")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки модели: {e}")

# Чёрный список
EXPLICIT_BAD_WORDS = [
    "фуфло", "хамство", "хамить", "грубить", "г*вно", "дерьмо", "идиот", "тупой",
    "мудак", "сука", "еб*н", "пизд", "бля", "херня", "долбоёб", "редиска", "скотина",
    "сволочь", "тварь", "ублюдок", "козёл", "осёл", "дебил", "кретин", "гнида", "падла"
]

def explicit_toxicity(text):
    text_low = text.lower()
    for word in EXPLICIT_BAD_WORDS:
        if word in text_low:
            return True
    return False

def analyze_with_neural(text):
    """Анализ отзыва с помощью нейросети"""
    sent = sentiment_pipeline(text)[0]
    tox = toxicity_pipeline(text)[0]
    is_toxic = tox['label'] == 'toxic' or explicit_toxicity(text)
    
    hidden = False
    if not is_toxic and _hidden_model is not None:
        result = _hidden_model(text)[0]
        hidden = result['label'] == 'LABEL_1'
    
    return 1 if (is_toxic or hidden) else 0

def analyze_with_heuristic(text):
    """Простая эвристика для сравнения"""
    text_low = text.lower()
    bad_phrases = ["фуфло", "хамство", "г*вно", "ужас", "кошмар", "позор", "обманули", "потеряли заказ"]
    for phrase in bad_phrases:
        if phrase in text_low:
            return 1
    if "спасибо за сервис" in text_low and ("потеряли" in text_low or "не пришёл" in text_low or "брак" in text_low):
        return 1
    return 0

# ========== СОЗДАНИЕ ТЕСТОВОГО НАБОРА ==========
TEST_FILE = "test_reviews.csv"
if not os.path.exists(TEST_FILE):
    print(f"📝 Создаю тестовый файл {TEST_FILE}...")
    test_data = [
        ("спасибо за сервис, всё отлично", 0),
        ("спасибо за сервис потеряли заказ", 1),
        ("отличные доски, рекомендую", 0),
        ("вы ужасная компания, обманули", 1),
        ("нормальное качество, но дороговато", 0),
        ("восхитительно! прислали брак", 1),
        ("быстрая доставка, спасибо", 0),
        ("полное фуфло, деньги на ветер", 1),
        ("ну вы даёте, уважаемые!", 1),
        ("хороший сервис, всем советую", 0),
        ("менеджеры хамят и грубят", 1),
        ("доска пришла кривая, но обменяли", 0),
        ("уникальные люди: потеряли заявку", 1),
        ("спасибо, всё чётко", 0),
    ]
    with open(TEST_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label"])
        for text, label in test_data:
            writer.writerow([text, label])
    print(f"✅ Создано {len(test_data)} тестовых отзывов")

# Загрузка тестовых отзывов
test_reviews = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_reviews.append((row['text'], int(row['true_label'])))

print(f"\n📋 Загружено {len(test_reviews)} тестовых отзывов")

# ========== ЗАПУСК A/B ТЕСТА ==========
print("\n" + "="*60)
print("🔬 ЗАПУСК A/B ТЕСТИРОВАНИЯ")
print("="*60)

neural_correct = 0
heuristic_correct = 0
neural_errors = []
heuristic_errors = []

for text, true_label in test_reviews:
    neural_pred = analyze_with_neural(text)
    heuristic_pred = analyze_with_heuristic(text)
    
    if neural_pred == true_label:
        neural_correct += 1
    else:
        neural_errors.append((text, true_label, neural_pred))
    
    if heuristic_pred == true_label:
        heuristic_correct += 1
    else:
        heuristic_errors.append((text, true_label, heuristic_pred))

neural_acc = neural_correct / len(test_reviews)
heuristic_acc = heuristic_correct / len(test_reviews)

# ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
print(f"\n📊 РЕЗУЛЬТАТЫ A/B ТЕСТА:")
print(f"┌─────────────────────┬────────────┬────────────┐")
print(f"│ Метод               │ Точность   │ Ошибок     │")
print(f"├─────────────────────┼────────────┼────────────┤")
print(f"│ Нейросеть           │ {neural_acc:.1%}        │ {len(neural_errors):2d}/{len(test_reviews)} │")
print(f"│ Эвристика (базовая) │ {heuristic_acc:.1%}        │ {len(heuristic_errors):2d}/{len(test_reviews)} │")
print(f"└─────────────────────┴────────────┴────────────┘")

improvement = (neural_acc - heuristic_acc) * 100
print(f"\n🚀 УЛУЧШЕНИЕ ТОЧНОСТИ: +{improvement:.1f}% в пользу нейросети")

# Детальные ошибки
if neural_errors:
    print("\n❌ Ошибки нейросети:")
    for text, true, pred in neural_errors[:3]:
        print(f"  • \"{text[:50]}...\" → верно: {true}, получили: {pred}")
if heuristic_errors:
    print("\n⚠️ Ошибки эвристики:")
    for text, true, pred in heuristic_errors[:4]:
        print(f"  • \"{text[:50]}...\" → верно: {true}, получили: {pred}")

# Сохранение отчёта
report = {
    "test_size": len(test_reviews),
    "neural_accuracy": neural_acc,
    "heuristic_accuracy": heuristic_acc,
    "improvement_percent": improvement,
    "neural_errors_count": len(neural_errors),
    "heuristic_errors_count": len(heuristic_errors),
    "neural_errors": [{"text": t, "true_label": tl, "predicted": p} for t, tl, p in neural_errors],
    "heuristic_errors": [{"text": t, "true_label": tl, "predicted": p} for t, tl, p in heuristic_errors]
}

with open("ab_test_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n✅ Отчёт сохранён в ab_test_report.json")
print("🏁 A/B тестирование завершено!")