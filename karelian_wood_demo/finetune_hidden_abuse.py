import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Параметры
MODEL_NAME = "cointegrated/rubert-tiny"
OUTPUT_DIR = "./model_hidden_abuse"
DATA_PATH = "data/hidden_abuse_dataset.csv"
TEST_SIZE = 0.2
EPOCHS = 7
LEARNING_RATE = 1.5e-5
BATCH_SIZE = 16
MAX_LEN = 128

# Проверка наличия данных
if not os.path.exists(DATA_PATH):
    print(f"❌ Файл {DATA_PATH} не найден.")
    exit(1)

# Загрузка данных
df = pd.read_csv(DATA_PATH)
df['label'] = df['label'].astype(int)
print(f"📊 Загружено {len(df)} примеров. Распределение:\n{df['label'].value_counts()}")

# Разделение на train/validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=TEST_SIZE,
    random_state=42,
    stratify=df['label']
)

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Создание датасетов
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Загрузка модели НА CPU
device = torch.device('cpu')
print(f"🖥️ Используется устройство: {device} (MPS отключён для стабильности)")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Обучение
print("\n🚀 НАЧАЛО ДООБУЧЕНИЯ...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    
    # Валидация
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    acc = (np.array(predictions) == np.array(val_labels)).mean()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

# Сохранение модели
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Модель сохранена в {OUTPUT_DIR}")

# Метрики
print("\n" + "="*60)
print("📊 МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
print("="*60)

report = classification_report(val_labels, predictions, target_names=['Безопасно (0)', 'Скрытое оскорбление (1)'])
print(report)

cm = confusion_matrix(val_labels, predictions)
print("\n📈 Матрица ошибок:")
print("               Предсказано")
print("              0       1")
print(f"Реально 0    {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"Реально 1    {cm[1,0]:3d}    {cm[1,1]:3d}")

# Сохранение метрик
metrics = {
    "accuracy": float(acc),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(val_labels, predictions, target_names=['Безопасно', 'Скрытое оскорбление'], output_dict=True),
    "num_examples": len(df),
    "num_train": len(train_texts),
    "num_val": len(val_texts)
}
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n✅ Метрики сохранены в model_metrics.json")

# Демонстрация
print("\n📝 ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
model.eval()
for text, true in zip(val_texts[:5], val_labels[:5]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    # Каждый input на CPU отдельно
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    status = "✅ Безопасно" if pred == 0 else "⚠️ Скрытое оскорбление"
    print(f"  Текст: {text[:50]}...")
    print(f"    Верно: {true} | Предсказано: {pred} | {status}\n")