from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    
from django.apps import AppConfig

class GoldAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gold_app'

    def ready(self):
        """Initialize model when app is ready"""
        try:
            from .predictions import GoldPriceModel
            self.gold_model = GoldPriceModel()
            self.gold_model.train_models()
            print("Gold price prediction model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize prediction model: {str(e)}")