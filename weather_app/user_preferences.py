import json
import os
from datetime import datetime

class UserPreferencesCollector:
    def __init__(self, data_file='ml_models/user_preferences.json'):
        self.data_file = data_file
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        """Загрузка предпочтений пользователей"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_preference(self, weather_data, recommended_items, selected_items, feedback_type='good'):
        """
        Сохранение выбора пользователя
        
        Args:
            weather_data: данные о погоде
            recommended_items: рекомендованные элементы
            selected_items: выбранные пользователем элементы
            feedback_type: тип обратной связи ('good' или 'bad')
        """
        preference = {
            'timestamp': datetime.now().isoformat(),
            'weather': weather_data,
            'recommended': recommended_items,
            'selected': selected_items,
            'feedback_type': feedback_type,
            'match_score': self._calculate_match_score(recommended_items, selected_items)
        }
        
        self.preferences.append(preference)
        
        # Ограничиваем размер данных (сохраняем последние 1000 записей)
        if len(self.preferences) > 1000:
            self.preferences = self.preferences[-1000:]
        
        # Сохраняем в файл
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)
        
        print(f"Предпочтение сохранено (всего: {len(self.preferences)})")
        return len(self.preferences)
    
    def _calculate_match_score(self, recommended, selected):
        """Вычисление совпадения рекомендаций с выбором"""
        if not recommended or not selected:
            return 0
        
        recommended_set = set(recommended)
        selected_set = set(selected)
        
        if not recommended_set:
            return 0
        
        matches = len(recommended_set.intersection(selected_set))
        return matches / len(recommended_set)
    
    def get_training_data(self, min_records=50):
        """Получение данных для обучения"""
        if len(self.preferences) < min_records:
            print(f"Недостаточно данных для обучения ({len(self.preferences)} записей, требуется {min_records})")
            return None
        
        # Преобразование в формат для обучения (совместимо с ml_wardrobe.py)
        X = []
        y = []
        
        for pref in self.preferences:
            try:
                weather = pref['weather']
                timestamp = datetime.fromisoformat(pref['timestamp'])
                
                # Признаки совместимые с ml_wardrobe.py
                features = [
                    float(weather.get('temp', 20)),
                    float(weather.get('feels_like', weather.get('temp', 20))),
                    float(weather.get('humidity', 50)),
                    float(weather.get('wind_speed', 0)),
                    float(weather.get('pressure', 1013)),
                    float(weather.get('visibility', 10000)),
                    1 if any(x in weather.get('weather_main', '').lower() for x in ['rain', 'drizzle']) else 0,
                    1 if any(x in weather.get('weather_main', '').lower() for x in ['snow', 'sleet']) else 0,
                    1 if 'clear' in weather.get('weather_main', '').lower() else 0,
                    1 if any(x in weather.get('weather_main', '').lower() for x in ['clouds', 'overcast']) else 0,
                    timestamp.hour,
                    timestamp.month,
                    # Сезон (зима)
                    1 if timestamp.month in [12, 1, 2] else 0,
                    # Сезон (весна)
                    1 if timestamp.month in [3, 4, 5] else 0,
                    # Сезон (лето)
                    1 if timestamp.month in [6, 7, 8] else 0,
                    # Сезон (осень)
                    1 if timestamp.month in [9, 10, 11] else 0,
                ]
                
                X.append(features)
                
                # Целевая переменная: насколько совпали рекомендации
                # Используем match_score или feedback_type
                if pref.get('feedback_type') == 'good':
                    y.append(pref['match_score'])
                else:
                    y.append(0)  # Если плохая обратная связь
                    
            except Exception as e:
                print(f"Ошибка обработки предпочтения: {e}")
                continue
        
        if len(X) < min_records:
            print(f"Недостаточно качественных данных для обучения: {len(X)} записей")
            return None
        
        return X, y
    
    def get_statistics(self):
        """Статистика по предпочтениям (совместимость с app.py)"""
        if not self.preferences:
            return {
                'total_preferences': 0,
                'good_feedback': 0,
                'avg_match_score': "0%",
                'data_quality': 'Нет сохраненных предпочтений'
            }
        
        total = len(self.preferences)
        good_feedback = sum(1 for p in self.preferences if p.get('feedback_type') == 'good')
        avg_match = sum(p.get('match_score', 0) for p in self.preferences) / total
        
        return {
            'total_preferences': total,
            'good_feedback': good_feedback,
            'avg_match_score': f"{avg_match:.1%}",
            'data_quality': 'Отличная' if total >= 100 else f"Собирается ({total}/100)"
        }
    
    def get_preferences_for_retraining(self):
        """Получение предпочтений для переобучения модели"""
        return self.preferences
    
    def clear_preferences(self):
        """Очистка всех предпочтений"""
        self.preferences = []
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("Все предпочтения очищены")
    
    def get_user_insights(self):
        """Получение аналитики по пользовательским предпочтениям"""
        if not self.preferences:
            return {"message": "Нет данных для анализа"}
        
        # Анализ самых популярных элементов
        all_selected = []
        for pref in self.preferences:
            all_selected.extend(pref.get('selected', []))
        
        from collections import Counter
        item_counts = Counter(all_selected)
        
        # Самые популярные элементы
        top_items = item_counts.most_common(10)
        
        # Анализ погодных условий для популярных элементов
        weather_patterns = {}
        for item, count in top_items:
            weather_patterns[item] = {
                'count': count,
                'avg_temp': 0,
                'common_weather': []
            }
            
            # Собираем данные о погоде для этого элемента
            temps = []
            weather_types = []
            for pref in self.preferences:
                if item in pref.get('selected', []):
                    temps.append(pref['weather'].get('temp', 20))
                    weather_types.append(pref['weather'].get('weather_main', ''))
            
            if temps:
                weather_patterns[item]['avg_temp'] = sum(temps) / len(temps)
                weather_patterns[item]['common_weather'] = Counter(weather_types).most_common(3)
        
        return {
            'total_preferences': len(self.preferences),
            'top_selected_items': top_items,
            'weather_patterns': weather_patterns,
            'avg_match_score': sum(p.get('match_score', 0) for p in self.preferences) / len(self.preferences)
        }