# stock_app/urls.py - 子应用路由文件

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/login/', views.login_tushare, name='login'),
    path('api/save_token/', views.save_token, name='save_token'),
    path('api/select/', views.dual_verify_stocks, name='select'),
    path('api/status/', views.system_status, name='status'),
    path('api/kline/', views.get_kline_data, name='kline'),
]