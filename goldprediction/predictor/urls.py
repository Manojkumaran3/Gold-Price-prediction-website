from django.urls import path
from . import views

urlpatterns = [
    path('api/market-data/', views.market_data, name='market_data'),
    path('api/predict/', views.predict_price, name='predict_price'),
    
    path("",views.login_view,name="login"),
    path("base",views.base,name="base"),
    path("about",views.about,name="about"),
    path("index",views.index,name="index"),
    path("contact",views.contact_view,name="contact"),
    path("service",views.service,name="service"),
    path("team",views.team,name="team"),
   
    path("register",views.register_view,name="register"),

    path('pred', views.pred, name='pred')
]

'''from django.urls import path
from .views import prediction_page, predict_price

urlpatterns = [
    path('', prediction_page, name='gold_prediction'),
    path('api/predict/', predict_price, name='gold_prediction_api'),
]'''

