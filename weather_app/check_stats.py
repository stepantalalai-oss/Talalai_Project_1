# check_stats.py
import requests
import json

print("🔍 Проверка статистики модели...")
print()

try:
    # Отправляем запрос к серверу
    response = requests.get('http://localhost:5000/ml_info')
    
    if response.status_code == 200:
        data = response.json()
        
        print('📊 ОБНОВЛЁННАЯ СТАТИСТИКА:')
        print('=' * 40)
        
        # Информация о модели
        ml_model = data.get('ml_model', {})
        print(f'Модель: {ml_model.get("model_type")}')
        print(f'Точность: {ml_model.get("accuracy", "не указана")}')
        print(f'Статус: {ml_model.get("status")}')
        print()
        
        # Пользовательские предпочтения
        user_prefs = data.get('user_preferences', {})
        print(f'Записей пользователей: {user_prefs.get("total_preferences")}')
        print(f'Совпадение: {user_prefs.get("avg_match_score")}')
        print(f'Качество данных: {user_prefs.get("data_quality")}')
        
        # Системная информация
        system_info = data.get('system_info', {})
        print(f'Признаков: {system_info.get("total_features")}')
        print(f'Последнее переобучение: {system_info.get("last_retrained")}')
        
    else:
        print(f'❌ Ошибка сервера: {response.status_code}')
        
except requests.exceptions.ConnectionError:
    print('❌ Сервер не запущен!')
    print('Запустите: python app.py')
except Exception as e:
    print(f'❌ Ошибка: {e}')