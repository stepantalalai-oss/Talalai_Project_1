# force_retrain.py
from ml_wardrobe import WardrobeMLRecommender
from user_preferences import UserPreferencesCollector
import json

print("🔄 ПРИНУДИТЕЛЬНОЕ ПЕРЕОБУЧЕНИЕ")
print("=" * 40)

# Загружаем данные
collector = UserPreferencesCollector()
prefs = collector.preferences
print(f"📊 Записей для обучения: {len(prefs)}")

if len(prefs) < 10:
    print("❌ Слишком мало данных")
else:
    # Переобучаем модель
    print("🧠 Запуск переобучения...")
    ml = WardrobeMLRecommender()
    accuracy = ml.retrain_with_user_preferences(prefs)
    
    print(f"✅ Переобучение завершено!")
    print(f"🎯 Точность: {accuracy:.1%}")
    
    # Обновляем retrain_log.json
    log_entry = {
        'timestamp': '2024-01-30T15:45:00',
        'records_used': len(prefs),
        'accuracy': float(accuracy),
        'total_records': len(prefs)
    }
    
    with open('ml_models/retrain_log.json', 'a', encoding='utf-8') as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write('\n')
    
    print("📁 Лог обновлён")