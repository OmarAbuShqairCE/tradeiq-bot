from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from datetime import timezone
from dotenv import load_dotenv
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available. Install guide: https://github.com/mrjbq7/ta-lib")
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("⚠️ TPOT (AutoML) not available. Install with: pip install tpot")
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import cv2
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # استخدام backend غير تفاعلي لتجنب مشاكل tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import warnings
import logging
warnings.filterwarnings('ignore')

# تعطيل سجلات Flask الزائدة
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)


class AITradingSystem:
    """نظام الذكاء الاصطناعي المتقدم للتداول"""
    
    def __init__(self):
        self.ml_models = {}
        self.neural_models = {}
        self.time_series_models = {}
        self.reinforcement_models = {}
        self.advanced_models = {}  # XGBoost, LightGBM, CatBoost, AutoML
        self.scalers = {}
        self.label_encoders = {}
        self.performance_history = []
        self.model_metrics = {}
        self.q_table = {}  # إضافة q_table للتعلم المعزز
        
        # إعدادات النماذج
        self.model_config = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1},
            'neural_network': {'hidden_layers': (100, 50), 'activation': 'relu'},
            'lstm': {'units': 50, 'dropout': 0.2, 'epochs': 50},
            'cnn': {'filters': 32, 'kernel_size': 3, 'epochs': 30}
        }
        
        # تحميل النماذج المحفوظة
        self.load_saved_models()
    
    def load_saved_models(self):
        """تحميل النماذج المحفوظة"""
        try:
            if os.path.exists('ai_models.pkl'):
                with open('ai_models.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    self.ml_models = saved_data.get('ml_models', {})
                    self.neural_models = saved_data.get('neural_models', {})
                    self.scalers = saved_data.get('scalers', {})
                    self.model_metrics = saved_data.get('metrics', {})
                print("✅ تم تحميل النماذج المحفوظة بنجاح")
        except Exception as e:
            print(f"⚠️ خطأ في تحميل النماذج: {e}")
    
    def save_models(self):
        """حفظ النماذج المدربة"""
        try:
            save_data = {
                'ml_models': self.ml_models,
                'neural_models': self.neural_models,
                'advanced_models': self.advanced_models,
                'scalers': self.scalers,
                'metrics': self.model_metrics
            }
            with open('ai_models.pkl', 'wb') as f:
                pickle.dump(save_data, f)
            print("✅ تم حفظ النماذج بنجاح")
        except Exception as e:
            print(f"❌ خطأ في حفظ النماذج: {e}")
    
    def prepare_training_data(self, price_data, indicators_data):
        """إعداد بيانات التدريب"""
        try:
            # دمج بيانات الأسعار والمؤشرات
            features = []
            labels = []
            
            for i in range(len(price_data) - 1):
                feature_vector = []
                
                # إضافة بيانات الأسعار (4 قيم ثابتة)
                current_price = price_data[i]
                next_price = price_data[i + 1]
                
                feature_vector.extend([
                    current_price['open'],
                    current_price['high'],
                    current_price['low'],
                    current_price['close']
                ])
                
                # إضافة ميزات فنية متقدمة
                # حساب مؤشرات إضافية من بيانات الأسعار
                if i >= 14:  # نحتاج 14 شمعة للمؤشرات المتقدمة
                    # Price momentum
                    price_momentum = (current_price['close'] - price_data[i-5]['close']) / price_data[i-5]['close']
                    feature_vector.append(price_momentum)
                    
                    # Volatility (التقلب)
                    recent_closes = [price_data[j]['close'] for j in range(max(0, i-14), i+1)]
                    volatility = np.std(recent_closes) if len(recent_closes) > 1 else 0
                    feature_vector.append(volatility)
                    
                    # High-Low range
                    hl_range = (current_price['high'] - current_price['low']) / current_price['close']
                    feature_vector.append(hl_range)
                    
                    # TA-Lib مؤشرات متقدمة (إذا كان متاحاً)
                    if TALIB_AVAILABLE:
                        try:
                            closes = np.array([price_data[j]['close'] for j in range(max(0, i-14), i+1)])
                            highs = np.array([price_data[j]['high'] for j in range(max(0, i-14), i+1)])
                            lows = np.array([price_data[j]['low'] for j in range(max(0, i-14), i+1)])
                            
                            # RSI
                            rsi = talib.RSI(closes, timeperiod=14)
                            feature_vector.append(rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50)
                            
                            # MACD
                            macd, signal, hist = talib.MACD(closes)
                            feature_vector.append(macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0)
                            
                            # ADX (Average Directional Index)
                            adx = talib.ADX(highs, lows, closes, timeperiod=14)
                            feature_vector.append(adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 0)
                            
                            # CCI (Commodity Channel Index)
                            cci = talib.CCI(highs, lows, closes, timeperiod=14)
                            feature_vector.append(cci[-1] if len(cci) > 0 and not np.isnan(cci[-1]) else 0)
                        except Exception:
                            feature_vector.extend([50, 0, 0, 0])  # قيم افتراضية
                    else:
                        feature_vector.extend([50, 0, 0, 0])  # قيم افتراضية
                else:
                    feature_vector.extend([0, 0, 0, 50, 0, 0, 0])  # 7 ميزات فنية
                
                # إضافة المؤشرات (تحديد عدد ثابت)
                indicator_count = 0
                max_indicators = 3  # زيادة عدد المؤشرات
                
                for indicator_name, indicator_data in indicators_data.items():
                    if indicator_count >= max_indicators:
                        break
                    if indicator_data and len(indicator_data) > i:
                        indicator_value = self._extract_indicator_value(indicator_data[i])
                        if indicator_value is not None:
                            feature_vector.append(indicator_value)
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.append(0)
                    indicator_count += 1
                
                # التأكد من أن عدد الميزات ثابت (4 أسعار + 7 ميزات فنية + 3 مؤشرات = 14)
                while len(feature_vector) < 14:
                    feature_vector.append(0)
                
                # حساب العلامة (النتيجة)
                price_change = (next_price['close'] - current_price['close']) / current_price['close']
                if price_change > 0.001:  # ارتفاع أكثر من 0.1%
                    label = 1  # BUY
                elif price_change < -0.001:  # انخفاض أكثر من 0.1%
                    label = 0  # SELL
                else:
                    label = 2  # HOLD
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
        except Exception as e:
            print(f"❌ خطأ في إعداد بيانات التدريب: {e}")
            return None, None
    
    def _extract_indicator_value(self, indicator_data):
        """استخراج قيمة المؤشر"""
        if isinstance(indicator_data, dict):
            for key in ['value', 'close', 'sma', 'ema', 'rsi', 'macd']:
                if key in indicator_data:
                    try:
                        return float(indicator_data[key])
                    except:
                        continue
        return None
    
    def train_ml_models(self, features, labels):
        """تدريب نماذج Machine Learning"""
        try:
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['main'] = scaler
            
            # تدريب Random Forest
            rf_model = RandomForestClassifier(**self.model_config['random_forest'])
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            self.ml_models['random_forest'] = rf_model
            self.model_metrics['random_forest'] = {'accuracy': rf_accuracy}
            
            # تدريب Gradient Boosting
            gb_model = GradientBoostingClassifier(**self.model_config['gradient_boosting'])
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            self.ml_models['gradient_boosting'] = gb_model
            self.model_metrics['gradient_boosting'] = {'accuracy': gb_accuracy}
            
            # تدريب Neural Network
            nn_model = MLPClassifier(**self.model_config['neural_network'])
            nn_model.fit(X_train_scaled, y_train)
            nn_pred = nn_model.predict(X_test_scaled)
            nn_accuracy = accuracy_score(y_test, nn_pred)
            self.ml_models['neural_network'] = nn_model
            self.model_metrics['neural_network'] = {'accuracy': nn_accuracy}
            
            # تدريب XGBoost (إذا كان متاحاً)
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        objective='multi:softmax',
                        num_class=len(np.unique(y_train)),
                        random_state=42,
                        eval_metric='mlogloss'
                    )
                    xgb_model.fit(X_train_scaled, y_train)
                    xgb_pred = xgb_model.predict(X_test_scaled)
                    xgb_accuracy = accuracy_score(y_test, xgb_pred)
                    self.advanced_models['xgboost'] = xgb_model
                    self.model_metrics['xgboost'] = {'accuracy': xgb_accuracy}
                    print(f"✅ XGBoost: {xgb_accuracy:.3f}")
                except Exception as e:
                    print(f"⚠️ خطأ في تدريب XGBoost: {e}")
            
            # تدريب LightGBM (إذا كان متاحاً)
            if LIGHTGBM_AVAILABLE:
                try:
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1
                    )
                    lgb_model.fit(X_train_scaled, y_train)
                    lgb_pred = lgb_model.predict(X_test_scaled)
                    lgb_accuracy = accuracy_score(y_test, lgb_pred)
                    self.advanced_models['lightgbm'] = lgb_model
                    self.model_metrics['lightgbm'] = {'accuracy': lgb_accuracy}
                    print(f"✅ LightGBM: {lgb_accuracy:.3f}")
                except Exception as e:
                    print(f"⚠️ خطأ في تدريب LightGBM: {e}")
            
            # تدريب SVM (للبيانات الصغيرة فقط)
            if len(X_train_scaled) < 1000:
                try:
                    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
                    svm_model.fit(X_train_scaled, y_train)
                    svm_pred = svm_model.predict(X_test_scaled)
                    svm_accuracy = accuracy_score(y_test, svm_pred)
                    self.ml_models['svm'] = svm_model
                    self.model_metrics['svm'] = {'accuracy': svm_accuracy}
                    print(f"✅ SVM: {svm_accuracy:.3f}")
                except Exception as e:
                    print(f"⚠️ خطأ في تدريب SVM: {e}")
            
            # تدريب CatBoost (إذا كان متاحاً)
            if CATBOOST_AVAILABLE:
                try:
                    cat_model = cb.CatBoostClassifier(
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        loss_function='MultiClass',
                        random_seed=42,
                        verbose=False
                    )
                    cat_model.fit(X_train_scaled, y_train)
                    cat_pred = cat_model.predict(X_test_scaled)
                    cat_accuracy = accuracy_score(y_test, cat_pred)
                    self.advanced_models['catboost'] = cat_model
                    self.model_metrics['catboost'] = {'accuracy': cat_accuracy}
                    print(f"✅ CatBoost: {cat_accuracy:.3f}")
                except Exception as e:
                    print(f"⚠️ خطأ في تدريب CatBoost: {e}")
            
            # تدريب AutoML (TPOT) - فقط للبيانات الصغيرة ولمرة واحدة
            if TPOT_AVAILABLE and len(X_train_scaled) < 500 and 'automl' not in self.advanced_models:
                try:
                    print("🤖 جاري تدريب AutoML (TPOT) - قد يستغرق بضع دقائق...")
                    tpot_model = TPOTClassifier(
                        generations=3,
                        population_size=10,
                        cv=3,
                        random_state=42,
                        verbosity=0,
                        max_time_mins=2,
                        n_jobs=1
                    )
                    tpot_model.fit(X_train_scaled, y_train)
                    tpot_pred = tpot_model.predict(X_test_scaled)
                    tpot_accuracy = accuracy_score(y_test, tpot_pred)
                    self.advanced_models['automl'] = tpot_model
                    self.model_metrics['automl'] = {'accuracy': tpot_accuracy}
                    print(f"✅ AutoML (TPOT): {tpot_accuracy:.3f}")
                except Exception as e:
                    print(f"⚠️ خطأ في تدريب AutoML: {e}")
            
            print(f"✅ تم تدريب نماذج ML - RF: {rf_accuracy:.3f}, GB: {gb_accuracy:.3f}, NN: {nn_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نماذج ML: {e}")
            return False
    
    def train_lstm_model(self, features, labels):
        """تدريب نموذج LSTM للسلاسل الزمنية"""
        try:
            # إعادة تشكيل البيانات للـ LSTM
            sequence_length = 10
            X_lstm, y_lstm = [], []
            
            for i in range(sequence_length, len(features)):
                X_lstm.append(features[i-sequence_length:i])
                y_lstm.append(labels[i])
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_lstm, y_lstm, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler_lstm = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = scaler_lstm.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_lstm.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            self.scalers['lstm'] = scaler_lstm
            
            # بناء نموذج LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[-1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            # تدريب النموذج
            model.fit(X_train_scaled, y_train, 
                     epochs=self.model_config['lstm']['epochs'], 
                     batch_size=32, 
                     validation_data=(X_test_scaled, y_test),
                     verbose=0)
            
            # تقييم النموذج
            lstm_loss, lstm_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.neural_models['lstm'] = model
            self.model_metrics['lstm'] = {'accuracy': lstm_accuracy, 'loss': lstm_loss}
            
            print(f"✅ تم تدريب نموذج LSTM - دقة: {lstm_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نموذج LSTM: {e}")
            return False
    
    def train_cnn_model(self, features, labels):
        """تدريب نموذج CNN للرؤية الحاسوبية"""
        try:
            # تحويل البيانات إلى تنسيق مناسب للـ CNN
            sequence_length = 20
            X_cnn, y_cnn = [], []
            
            for i in range(sequence_length, len(features)):
                X_cnn.append(features[i-sequence_length:i])
                y_cnn.append(labels[i])
            
            X_cnn = np.array(X_cnn)
            y_cnn = np.array(y_cnn)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_cnn, y_cnn, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler_cnn = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = scaler_cnn.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_cnn.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            self.scalers['cnn'] = scaler_cnn
            
            # بناء نموذج CNN
            model = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=(sequence_length, X_train.shape[-1])),
                MaxPooling1D(2),
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            # تدريب النموذج
            model.fit(X_train_scaled, y_train, 
                     epochs=self.model_config['cnn']['epochs'], 
                     batch_size=32, 
                     validation_data=(X_test_scaled, y_test),
                     verbose=0)
            
            # تقييم النموذج
            cnn_loss, cnn_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.neural_models['cnn'] = model
            self.model_metrics['cnn'] = {'accuracy': cnn_accuracy, 'loss': cnn_loss}
            
            print(f"✅ تم تدريب نموذج CNN - دقة: {cnn_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نموذج CNN: {e}")
            return False
    
    def train_transformer_model(self, features, labels):
        """تدريب نموذج Transformer للسلاسل الزمنية"""
        try:
            sequence_length = 15
            X_trans, y_trans = [], []
            
            for i in range(sequence_length, len(features)):
                X_trans.append(features[i-sequence_length:i])
                y_trans.append(labels[i])
            
            X_trans = np.array(X_trans)
            y_trans = np.array(y_trans)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_trans, y_trans, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler_trans = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = scaler_trans.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_trans.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            self.scalers['transformer'] = scaler_trans
            
            # بناء نموذج Transformer
            inputs = tf.keras.Input(shape=(sequence_length, X_train.shape[-1]))
            
            # Multi-Head Attention layer
            attention_output = MultiHeadAttention(
                num_heads=4,
                key_dim=X_train.shape[-1],
                dropout=0.1
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
            
            # Feed Forward Network
            ffn_output = Dense(64, activation='relu')(attention_output)
            ffn_output = Dropout(0.1)(ffn_output)
            ffn_output = Dense(X_train.shape[-1])(ffn_output)
            
            # Add & Norm
            ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
            
            # Global pooling
            pooled = GlobalAveragePooling1D()(ffn_output)
            
            # Classification head
            outputs = Dense(32, activation='relu')(pooled)
            outputs = Dropout(0.2)(outputs)
            outputs = Dense(3, activation='softmax')(outputs)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # تدريب النموذج
            model.fit(
                X_train_scaled, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                verbose=0
            )
            
            # تقييم النموذج
            trans_loss, trans_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.neural_models['transformer'] = model
            self.model_metrics['transformer'] = {'accuracy': trans_accuracy, 'loss': trans_loss}
            
            print(f"✅ تم تدريب نموذج Transformer - دقة: {trans_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نموذج Transformer: {e}")
            return False
    
    def predict_with_ensemble(self, features):
        """التنبؤ باستخدام مجموعة النماذج"""
        try:
            predictions = []
            probabilities = []

            # تحميل النماذج المحفوظة تلقائياً إذا كانت الذاكرة فارغة
            if not self.ml_models and not self.neural_models:
                try:
                    self.load_saved_models()
                except Exception:
                    pass

            # اختيار الـ scaler المتوافق مع عدد الميزات
            feature_len = len(features)
            chosen_scaler = None
            for name, sc in self.scalers.items():
                try:
                    n = getattr(sc, 'n_features_in_', None)
                    if n is None and hasattr(sc, 'mean_'):
                        n = sc.mean_.shape[0]
                    if n == feature_len:
                        chosen_scaler = sc
                        break
                except Exception:
                    continue

            # تطبيع البيانات إذا وُجد scaler مطابق
            if chosen_scaler is not None:
                features_scaled = chosen_scaler.transform([features])
            else:
                features_scaled = [features]

            # تنبؤات نماذج ML مع توحيد شكل الاحتمالات إلى 3 فئات [SELL, BUY, HOLD]
            all_models = {**self.ml_models, **self.advanced_models}
            for model_name, model in all_models.items():
                # تخطِ أي نموذج لا يطابق أبعاد الميزات
                try:
                    n_in = getattr(model, 'n_features_in_', None)
                    if n_in is not None and n_in != feature_len:
                        continue
                except Exception:
                    pass

                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(features_scaled)[0]
                    # توحيد عدد الفئات إلى 3
                    unified = [0.0, 0.0, 0.0]  # SELL=0, BUY=1, HOLD=2
                    try:
                        classes = list(getattr(model, 'classes_', []))
                        if classes:
                            # املأ الاحتمالات حسب الفئات الموجودة
                            for idx, cls in enumerate(classes):
                                if cls in [0, 1, 2]:
                                    unified[int(cls)] = float(pred_proba[idx]) if idx < len(pred_proba) else 0.0
                        else:
                            # في حال عدم توفر classes_، افترض نفس الترتيب
                            for i in range(min(3, len(pred_proba))):
                                unified[i] = float(pred_proba[i])
                    except Exception:
                        # أي خطأ، استخدم القيم كما هي مع حشو إلى 3
                        for i in range(min(3, len(pred_proba))):
                            unified[i] = float(pred_proba[i])

                    # إعادة التطبيع لضمان أن المجموع <= 1 (اختياري)
                    s = sum(unified)
                    if s > 0:
                        unified = [u / s for u in unified]

                    pred = int(np.argmax(unified))
                    predictions.append(pred)
                    # إعطاء وزن أعلى للنماذج المتقدمة
                    if model_name in ['catboost', 'automl']:
                        weight = 2.0  # أعلى وزن لـ CatBoost و AutoML
                    elif model_name in ['xgboost', 'lightgbm']:
                        weight = 1.5
                    else:
                        weight = 1.0
                    probabilities.append([u * weight for u in unified])

            # تنبؤات نماذج Neural Networks - معطلة مؤقتاً بسبب مشاكل في البيانات
            # LSTM و CNN يحتاجان إلى تسلسلات زمنية كاملة
            # if model_name == 'lstm':
            #     ...
            # elif model_name == 'cnn':
            #     ...

            # حساب المتوسط المرجح للتنبؤات
            if probabilities:
                avg_probabilities = np.mean(probabilities, axis=0)
                final_prediction = np.argmax(avg_probabilities)
                confidence = np.max(avg_probabilities) * 100

                # تحويل التوقع إلى نص
                signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
                signal = signal_map.get(final_prediction, 'HOLD')

                return {
                    'signal': signal,
                    'confidence': round(confidence, 2),
                    'probabilities': {
                        'BUY': round(avg_probabilities[1] * 100, 2),
                        'SELL': round(avg_probabilities[0] * 100, 2),
                        'HOLD': round(avg_probabilities[2] * 100, 2)
                    },
                    'model_predictions': len(predictions),
                    'ensemble_used': True
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                    'model_predictions': 0,
                    'ensemble_used': False
                }
        except Exception as e:
            print(f"❌ خطأ في التنبؤ: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                'model_predictions': 0,
                'ensemble_used': False
            }
    
    def analyze_chart_patterns(self, price_data):
        """تحليل الأنماط في الرسوم البيانية باستخدام الرؤية الحاسوبية"""
        try:
            # إنشاء رسم بياني
            plt.figure(figsize=(10, 6))
            prices = [p['close'] for p in price_data[-50:]]  # آخر 50 نقطة
            plt.plot(prices)
            plt.title('Price Chart Analysis')
            plt.xlabel('Time')
            plt.ylabel('Price')
            
            # حفظ الرسم كصورة
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # تحويل إلى صورة OpenCV
            img_data = img_buffer.getvalue()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # تحليل الأنماط
            patterns = self._detect_chart_patterns(img)
            
            return {
                'patterns_detected': patterns,
                'chart_analysis': True
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الأنماط: {e}")
            return {
                'patterns_detected': [],
                'chart_analysis': False
            }
    
    def _detect_chart_patterns(self, img):
        """كشف الأنماط في الرسم البياني"""
        try:
            # تحويل إلى تدرج رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # كشف الحواف
            edges = cv2.Canny(gray, 50, 150)
            
            # كشف الخطوط
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            patterns = []
            if lines is not None:
                # تحليل اتجاه الخطوط
                upward_lines = 0
                downward_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if -45 < angle < 45:  # خط أفقي
                        continue
                    elif angle > 0:  # خط صاعد
                        upward_lines += 1
                    else:  # خط هابط
                        downward_lines += 1
                
                # تحديد الأنماط
                if upward_lines > downward_lines * 1.5:
                    patterns.append('Uptrend')
                elif downward_lines > upward_lines * 1.5:
                    patterns.append('Downtrend')
                else:
                    patterns.append('Sideways')
            
            return patterns
        except Exception as e:
            print(f"❌ خطأ في كشف الأنماط: {e}")
            return []
    
    def reinforcement_learning_update(self, action, reward, state):
        """تحديث نموذج التعلم المعزز"""
        try:
            # تطبيق خوارزمية Q-Learning مبسطة
            if not hasattr(self, 'q_table'):
                self.q_table = {}
            
            state_key = str(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # تحديث قيمة Q
            learning_rate = 0.1
            discount_factor = 0.9
            
            old_value = self.q_table[state_key][action]
            self.q_table[state_key][action] = old_value + learning_rate * (reward - old_value)
            
            return True
        except Exception as e:
            print(f"❌ خطأ في تحديث التعلم المعزز: {e}")
            return False
    
    def get_reinforcement_prediction(self, state):
        """الحصول على تنبؤ من التعلم المعزز"""
        try:
            state_key = str(state)
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                best_action = max(q_values, key=q_values.get)
                confidence = abs(q_values[best_action]) / 10  # تحويل إلى نسبة مئوية
                return {
                    'signal': best_action,
                    'confidence': min(confidence * 100, 100),
                    'q_values': q_values
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'q_values': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                }
        except Exception as e:
            print(f"❌ خطأ في تنبؤ التعلم المعزز: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'q_values': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            }
    
    def get_performance_metrics(self):
        """الحصول على مقاييس الأداء"""
        return {
            'model_metrics': self.model_metrics,
            'performance_history': self.performance_history[-10:],  # آخر 10 قياسات
            'total_models': len(self.ml_models) + len(self.neural_models),
            'models_loaded': len(self.ml_models) > 0 or len(self.neural_models) > 0
        }


class TradingAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.twelvedata.com/time_series"
        self.is_running = False
        self.analysis_thread = None
        self.latest_results = {}
        
        # إضافة نظام الذكاء الاصطناعي
        self.ai_system = AITradingSystem()
        self.training_data = []
        self.ai_enabled = True
        
        # عداد الطلبات
        self.api_requests_count = 0
        self.api_requests_limit = 8  # الحد الأقصى للطلبات في الدقيقة
        self.last_reset_time = datetime.now()
        
        # نظام تقييم الصفقات
        self.trade_history = []
        self.pending_evaluations = []
        self.load_trade_history()
        
        # جميع أزواج العملات المتاحة
        self.available_pairs = [
            'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY', 'EUR/NZD', 'EUR/USD',
    'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/JPY', 'GBP/NZD', 'GBP/USD',
    'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/USD',
    'NZD/JPY', 'NZD/USD',
    'USD/CAD', 'USD/CHF', 'USD/JPY',
    'CAD/CHF', 'CAD/JPY',
    'CHF/JPY'
        ]
        
        # المؤشرات المتاحة حسب المجموعات
        self.available_indicators = {
            # Trend Indicators (مؤشرات الاتجاه)
            'trend': {
                'sma': 'Simple Moving Average',
                'ema': 'Exponential Moving Average', 
                'wma': 'Weighted Moving Average',
                'dema': 'Double Exponential Moving Average',
                'tema': 'Triple Exponential Moving Average'
                # ملاحظة: تم حذف t3, hma, kama لأنها غير مدعومة من TwelveData API
            },
            # Momentum Indicators (مؤشرات الزخم)
            'momentum': {
                'rsi': 'Relative Strength Index',
                'stoch': 'Stochastic Oscillator',
                'stochrsi': 'Stochastic RSI',
                'willr': 'Williams %R',
                'macd': 'MACD',
                'ppo': 'Percentage Price Oscillator',
                'adx': 'Average Directional Index',
                'cci': 'Commodity Channel Index',
                'mom': 'Momentum',
                'roc': 'Rate of Change'
            },
            # Volatility Indicators (مؤشرات التذبذب)
            'volatility': {
                'bbands': 'Bollinger Bands',
                'atr': 'Average True Range',
                'stdev': 'Standard Deviation',
                'donchian': 'Donchian Channels'
            },
            # Volume Indicators (مؤشرات الحجم)
            'volume': {
                'obv': 'On Balance Volume',
                'cmf': 'Chaikin Money Flow',
                'ad': 'Accumulation/Distribution',
                'mfi': 'Money Flow Index',
                'emv': 'Ease of Movement',
                'fi': 'Force Index'
            },
            # Price Indicators (مؤشرات السعر)
            'price': {
                'avgprice': 'Average Price',
                'medprice': 'Median Price',
                'typprice': 'Typical Price',
                'wcprice': 'Weighted Close Price'
            },
            # Misc / Other Indicators (مؤشرات أخرى)
            'misc': {
                'sar': 'Parabolic SAR',
                'ultosc': 'Ultimate Oscillator',
                'tsi': 'True Strength Index'
            }
        }
    
    def set_api_key(self, api_key):
        """تعيين مفتاح API"""
        self.api_key = api_key
    
    def _check_api_limit(self):
        """التحقق من حد الطلبات"""
        current_time = datetime.now()
        
        # إعادة تعيين العداد كل دقيقة
        if (current_time - self.last_reset_time).seconds >= 60:
            self.api_requests_count = 0
            self.last_reset_time = current_time
        
        return self.api_requests_count < self.api_requests_limit
    
    def _increment_api_count(self):
        """زيادة عداد الطلبات"""
        self.api_requests_count += 1
    
    def get_api_status(self):
        """الحصول على حالة API"""
        current_time = datetime.now()
        time_remaining = 60 - (current_time - self.last_reset_time).seconds
        
        return {
            'requests_used': self.api_requests_count,
            'requests_limit': self.api_requests_limit,
            'requests_remaining': self.api_requests_limit - self.api_requests_count,
            'time_remaining': max(0, time_remaining),
            'can_make_request': self._check_api_limit()
        }
    
    def load_trade_history(self):
        """تحميل تاريخ الصفقات"""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r', encoding='utf-8') as f:
                    self.trade_history = json.load(f)
                print(f"✅ تم تحميل {len(self.trade_history)} صفقة من التاريخ")
        except Exception as e:
            print(f"❌ خطأ في تحميل تاريخ الصفقات: {e}")
            self.trade_history = []
    
    def save_trade_history(self):
        """حفظ تاريخ الصفقات"""
        try:
            with open('trade_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.trade_history, f, ensure_ascii=False, indent=2)
            print("✅ تم حفظ تاريخ الصفقات")
        except Exception as e:
            print(f"❌ خطأ في حفظ تاريخ الصفقات: {e}")
    
    def add_trade_for_evaluation(self, pair, signal, entry_price, entry_time, exit_time, indicators_data, trade_type='real'):
        """إضافة صفقة للتقييم"""
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pair.replace('/', '_')}"
        
        # إنشاء ملاحظات ذكية باستخدام الذكاء الاصطناعي
        ai_notes = self.generate_ai_notes(pair, signal, entry_price, indicators_data)
        
        trade = {
            'id': trade_id,
            'pair': pair,
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'indicators_data': indicators_data,
            'trade_type': trade_type,  # 'real' or 'test'
            'status': 'pending',  # pending, successful, failed, cancelled
            'user_evaluation': None,
            'ai_notes': ai_notes,
            'user_notes': '',
            'created_at': datetime.now().isoformat()
        }
        
        self.pending_evaluations.append(trade)
        print(f"📝 تم إضافة صفقة {trade_type} {pair} للتقييم: {trade_id}")
        return trade_id
    
    def generate_ai_notes(self, pair, signal, entry_price, indicators_data):
        """إنشاء ملاحظات ذكية باستخدام الذكاء الاصطناعي"""
        try:
            notes = []
            
            # تحليل الإشارة
            if signal == 'CALL':
                notes.append("📈 إشارة صعود - توقع ارتفاع في السعر")
            else:
                notes.append("📉 إشارة هبوط - توقع انخفاض في السعر")
            
            # تحليل المؤشرات
            if indicators_data:
                if 'rsi' in indicators_data:
                    rsi = indicators_data['rsi']
                    if rsi < 30:
                        notes.append(f"🔴 RSI منخفض ({rsi:.1f}) - تشبع بيع محتمل")
                    elif rsi > 70:
                        notes.append(f"🟢 RSI مرتفع ({rsi:.1f}) - تشبع شراء محتمل")
                    else:
                        notes.append(f"⚖️ RSI متوازن ({rsi:.1f}) - سوق محايد")
                
                if 'macd' in indicators_data:
                    macd = indicators_data['macd']
                    if macd > 0:
                        notes.append(f"📊 MACD إيجابي ({macd:.4f}) - زخم صاعد")
                    else:
                        notes.append(f"📊 MACD سلبي ({macd:.4f}) - زخم هابط")
                
                # تحليل الأسعار
                if all(key in indicators_data for key in ['open', 'high', 'low', 'close']):
                    open_price = indicators_data['open']
                    close_price = indicators_data['close']
                    high_price = indicators_data['high']
                    low_price = indicators_data['low']
                    
                    price_change = ((close_price - open_price) / open_price) * 100
                    if price_change > 0.1:
                        notes.append(f"📈 تغير إيجابي في السعر (+{price_change:.2f}%)")
                    elif price_change < -0.1:
                        notes.append(f"📉 تغير سلبي في السعر ({price_change:.2f}%)")
                    
                    # تحليل التقلبات
                    volatility = ((high_price - low_price) / open_price) * 100
                    if volatility > 1:
                        notes.append(f"⚡ تقلبات عالية ({volatility:.2f}%) - سوق متقلب")
                    else:
                        notes.append(f"📊 تقلبات منخفضة ({volatility:.2f}%) - سوق مستقر")
            
            # تحليل الزوج
            if 'USD' in pair:
                notes.append("💵 زوج يحتوي على الدولار - تأثر بالأخبار الأمريكية")
            elif 'EUR' in pair:
                notes.append("🇪🇺 زوج أوروبي - تأثر بأخبار المنطقة الأوروبية")
            elif 'GBP' in pair:
                notes.append("🇬🇧 زوج بريطاني - تأثر بأخبار المملكة المتحدة")
            
            # نصيحة عامة
            notes.append("💡 نصيحة: راقب الأخبار الاقتصادية وتأكد من إدارة المخاطر")
            
            return " | ".join(notes)
            
        except Exception as e:
            print(f"❌ خطأ في إنشاء الملاحظات الذكية: {e}")
            return "ملاحظات ذكية غير متاحة حالياً"
    
    def evaluate_trade(self, trade_id, evaluation, notes="", user_notes=""):
        """تقييم صفقة"""
        try:
            # البحث عن الصفقة في القائمة المعلقة
            trade = None
            for i, t in enumerate(self.pending_evaluations):
                if t['id'] == trade_id:
                    trade = t
                    del self.pending_evaluations[i]
                    break
            
            if not trade:
                return False
            
            # تحديث الصفقة
            trade['user_evaluation'] = evaluation
            trade['notes'] = notes
            trade['user_notes'] = user_notes
            trade['evaluated_at'] = datetime.now().isoformat()
            
            if evaluation == 'cancelled':
                trade['status'] = 'cancelled'
                print(f"❌ تم إلغاء الصفقة {trade_id}")
            else:
                trade['status'] = 'successful' if evaluation == 'successful' else 'failed'
                # إضافة للتاريخ فقط إذا لم تكن ملغاة
                self.trade_history.append(trade)
                print(f"✅ تم تقييم الصفقة {trade_id}: {evaluation}")

                # تحديث التعلم المعزز بناءً على تقييم المستخدم
                try:
                    action = 'BUY' if trade.get('signal') == 'CALL' else 'SELL'
                    reward = 1 if evaluation == 'successful' else -1
                    # حالة مبسطة من OHLC + حتى 2 مؤشر كما في التدريب
                    indicators = trade.get('indicators_data', {})
                    state = [
                        indicators.get('open', 0),
                        indicators.get('high', 0),
                        indicators.get('low', 0),
                        indicators.get('close', 0)
                    ]
                    # إضافة حتى مؤشرين
                    indicator_count = 0
                    for key, value in indicators.items():
                        if key in ['open','high','low','close']:
                            continue
                        if indicator_count >= 2:
                            break
                        try:
                            state.append(float(value) if value is not None else 0)
                        except Exception:
                            state.append(0)
                        indicator_count += 1
                    while len(state) < 6:
                        state.append(0)
                    self.ai_system.reinforcement_learning_update(action, reward, state)
                except Exception as _:
                    pass
            
            # حفظ التاريخ
            self.save_trade_history()
            
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تقييم الصفقة: {e}")
            return False
    
    def get_pending_evaluations(self):
        """الحصول على الصفقات المعلقة للتقييم"""
        return self.pending_evaluations
    
    def get_trade_history(self, limit=50):
        """الحصول على تاريخ الصفقات"""
        return self.trade_history[-limit:]
    
    def get_trade_statistics(self):
        """الحصول على إحصائيات الصفقات"""
        # إحصائيات الصفقات الحقيقية
        real_trades = [t for t in self.trade_history if t.get('trade_type') == 'real']
        real_successful = len([t for t in real_trades if t.get('user_evaluation') == 'successful'])
        real_failed = len([t for t in real_trades if t.get('user_evaluation') == 'failed'])
        real_success_rate = round((real_successful / len(real_trades)) * 100, 2) if len(real_trades) > 0 else 0
        
        # إحصائيات الصفقات التجريبية
        test_trades = [t for t in self.trade_history if t.get('trade_type') == 'test']
        test_successful = len([t for t in test_trades if t.get('user_evaluation') == 'successful'])
        test_failed = len([t for t in test_trades if t.get('user_evaluation') == 'failed'])
        test_success_rate = round((test_successful / len(test_trades)) * 100, 2) if len(test_trades) > 0 else 0
        
        # إحصائيات الصفقات المعلقة
        pending_real = len([t for t in self.pending_evaluations if t.get('trade_type') == 'real'])
        pending_test = len([t for t in self.pending_evaluations if t.get('trade_type') == 'test'])
        
        # إحصائيات عامة
        total_trades = len(self.trade_history)
        total_successful = len([t for t in self.trade_history if t.get('user_evaluation') == 'successful'])
        total_failed = len([t for t in self.trade_history if t.get('user_evaluation') == 'failed'])
        total_success_rate = round((total_successful / total_trades) * 100, 2) if total_trades > 0 else 0
        
        return {
            # إحصائيات عامة
            'total_trades': total_trades,
            'successful_trades': total_successful,
            'failed_trades': total_failed,
            'success_rate': total_success_rate,
            'pending_evaluations': len(self.pending_evaluations),
            
            # إحصائيات الصفقات الحقيقية
            'real_trades': len(real_trades),
            'successful_real_trades': real_successful,
            'failed_real_trades': real_failed,
            'real_success_rate': real_success_rate,
            'pending_real_trades': pending_real,
            
            # إحصائيات الصفقات التجريبية
            'test_trades': len(test_trades),
            'successful_test_trades': test_successful,
            'failed_test_trades': test_failed,
            'test_success_rate': test_success_rate,
            'pending_test_trades': pending_test
        }
    
   
    #---------------------------------------------------
    def fetch_indicator_data(self, pair, indicator, interval='1min', **params):
        """جلب بيانات مؤشر من API TwelveData"""
        try:
            # التحقق من حد الطلبات
            if not self._check_api_limit():
                print(f"تم الوصول للحد الأقصى من الطلبات ({self.api_requests_limit})")
                return None
            
            # إعداد المعاملات الأساسية
            api_params = {
                'symbol': pair,
                'interval': interval,
                'apikey': self.api_key,
                **params
            }
            
            # بناء URL المؤشر
            indicator_url = f"https://api.twelvedata.com/{indicator}"
            
            # زيادة عداد الطلبات
            self._increment_api_count()
            print(f"طلب API #{self.api_requests_count}/{self.api_requests_limit}: {indicator} لـ {pair}")
            
            response = requests.get(indicator_url, params=api_params, timeout=10)
            
            if response.status_code == 404:
                print(f"⚠️ المؤشر {indicator} غير موجود في API (404) - تخطي")
                return None
            elif response.status_code != 200:
                print(f"خطأ HTTP عند جلب {indicator} لـ {pair}: حالة {response.status_code}")
                return None
            
            data = response.json()
            
            # تحقق من وجود خطأ في API
            if 'status' in data and data['status'] == 'error':
                error_msg = data.get('message', 'خطأ غير معروف')
                print(f"خطأ من API عند جلب {indicator} لـ {pair}: {error_msg}")
                
                # إذا كان الخطأ بسبب استنفاد الرصيد، انتظر دقيقة
                if 'API credits' in error_msg or 'limit' in error_msg:
                    print("انتظار 60 ثانية لتجنب استنفاد الرصيد...")
                    import time
                    time.sleep(60)
                
                return None
            
            if 'values' not in data or not data['values']:
                print(f"لا توجد بيانات لـ {indicator} لـ {pair}")
                return None
            
            # طباعة هيكل البيانات للتحقق
            if data['values'] and len(data['values']) > 0:
                print(f"هيكل بيانات {indicator}: {list(data['values'][0].keys())}")
            
            return data['values']
            
        except requests.exceptions.RequestException as e:
            print(f"خطأ في الاتصال عند جلب {indicator} لـ {pair}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"خطأ في فك JSON عند جلب {indicator} لـ {pair}: {str(e)}")
            return None
        except Exception as e:
            print(f"خطأ غير متوقع عند جلب {indicator} لـ {pair}: {str(e)}")
            return None

    def fetch_price_data(self, pair, interval='1min', outputsize=250):
        """جلب بيانات الأسعار الأساسية"""
        try:
            # التحقق من حد الطلبات
            if not self._check_api_limit():
                print(f"تم الوصول للحد الأقصى من الطلبات ({self.api_requests_limit})")
                return None
            
            params = {
                'symbol': pair,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            # زيادة عداد الطلبات
            self._increment_api_count()
            print(f"طلب API #{self.api_requests_count}/{self.api_requests_limit}: أسعار {pair}")
        
            response = requests.get(self.base_url, params=params, timeout=10)
        
            if response.status_code != 200:
                print(f"خطأ HTTP عند جلب {pair}: حالة {response.status_code}")
                return None
        
            data = response.json()
        
            if 'status' in data and data['status'] == 'error':
                print(f"خطأ من API عند جلب {pair}: {data.get('message', 'خطأ غير معروف')}")
                return None
        
            if 'values' not in data or not data['values']:
                print(f"لا توجد بيانات لـ {pair}")
                return None
        
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(hours=2)  # Convert to Palestine time (UTC+2)
            df = df.sort_values('datetime')
        
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # تأخير بسيط بعد جلب البيانات لتجنب استنفاد رصيد API
            import time
            time.sleep(2)
        
            return df
    
        except requests.exceptions.RequestException as e:
            print(f"خطأ في الاتصال عند جلب {pair}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"خطأ في فك JSON عند جلب {pair}: {str(e)}")
            return None
        except Exception as e:
            print(f"خطأ غير متوقع عند جلب {pair}: {str(e)}")
            return None

    #------------------------------------------------------
    def fetch_indicators_data(self, pair, selected_indicators, interval='1min'):
        """جلب بيانات المؤشرات المختارة من API"""
        indicators_data = {}
        
        for category, indicators in selected_indicators.items():
            if not indicators:  # تخطي المجموعات الفارغة
                continue
                
            for indicator in indicators:
                try:
                    # إعداد المعاملات الخاصة بكل مؤشر
                    params = self._get_indicator_params(indicator)
                    
                    # تخطي المؤشرات غير المدعومة
                    if params is None:
                        print(f"⏭️ تخطي {indicator} (غير مدعوم)")
                        continue
                    
                    # جلب بيانات المؤشر
                    data = self.fetch_indicator_data(pair, indicator, interval, **params)
                    
                    if data:
                        indicators_data[indicator] = data
                        print(f"تم جلب {indicator} لـ {pair}")
                    else:
                        print(f"فشل جلب {indicator} لـ {pair}")
                    
                    # تأخير أطول لتجنب استنفاد رصيد API
                    import time
                    time.sleep(10)  # 10 ثواني بين كل طلب (أكثر أماناً)
                        
                except Exception as e:
                    print(f"خطأ في جلب {indicator} لـ {pair}: {str(e)}")
                    continue
        
        return indicators_data

    def _get_indicator_params(self, indicator):
        """إرجاع المعاملات الافتراضية لكل مؤشر"""
        # قائمة المؤشرات غير المدعومة من TwelveData API
        unsupported_indicators = ['t3', 'hma', 'kama']  # هذه المؤشرات قد لا تكون مدعومة
        
        if indicator in unsupported_indicators:
            print(f"⚠️ المؤشر {indicator} غير مدعوم من TwelveData API")
            return None
        
        default_params = {
            'sma': {'time_period': 20},
            'ema': {'time_period': 20},
            'wma': {'time_period': 20},
            'dema': {'time_period': 20},
            'tema': {'time_period': 20},
            'rsi': {'time_period': 14},
            'stoch': {'fast_k_period': 14, 'slow_k_period': 3, 'slow_d_period': 3},
            'stochrsi': {'time_period': 14, 'fast_k_period': 3, 'fast_d_period': 3},
            'willr': {'time_period': 14},
            'macd': {'short_period': 12, 'long_period': 26, 'signal_period': 9},
            'ppo': {'short_period': 12, 'long_period': 26, 'signal_period': 9},
            'adx': {'time_period': 14},
            'cci': {'time_period': 14},
            'mom': {'time_period': 14},
            'roc': {'time_period': 14},
            'bbands': {'time_period': 20, 'series_type': 'close'},
            'atr': {'time_period': 14},
            'stdev': {'time_period': 20},
            'donchian': {'time_period': 20},
            'obv': {},
            'cmf': {'time_period': 20},
            'ad': {},
            'mfi': {'time_period': 14},
            'emv': {},
            'fi': {'time_period': 14},
            'avgprice': {},
            'medprice': {},
            'typprice': {},
            'wcprice': {},
            'sar': {'acceleration': 0.02, 'maximum': 0.2},
            'ultosc': {'time_period': 14},
            'tsi': {'time_period': 14}
        }
        
        return default_params.get(indicator, {})

    @staticmethod
    def safe_float(val):
        try:
            if pd.isna(val):
                return None
            return float(val)
        except:
            return None

    def _get_indicator_value(self, data, keys):
        """استخراج قيمة المؤشر من البيانات مع محاولة مفاتيح متعددة"""
        if not data or len(data) == 0:
            return None
            
        for key in keys:
            if key in data[0]:
                try:
                    return float(data[0][key])
                except:
                    continue
        return None

    def _get_indicator_values(self, data, keys, count=2):
        """استخراج قيم متعددة من المؤشر"""
        if not data or len(data) < count:
            return None
            
        values = []
        for i in range(count):
            for key in keys:
                if key in data[i]:
                    try:
                        values.append(float(data[i][key]))
                        break
                    except:
                        continue
            else:
                return None
        return values
        
    def analyze_trend_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الاتجاه"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Moving Averages
        for ma_type in ['sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'hma', 't3']:
            if ma_type in indicators_data:
                ma_data = indicators_data[ma_type]
                ma_values = self._get_indicator_values(ma_data, ['value', ma_type, 'sma', 'ema'])
                
                if ma_values and len(ma_values) >= 2:
                    try:
                        current_ma = ma_values[0]
                        prev_ma = ma_values[1]
                        current_price = float(price_data[-1]['close'])
                    except (ValueError, TypeError, KeyError, IndexError) as e:
                        print(f"⚠️ خطأ في قراءة بيانات {ma_type}: {e}")
                        continue
                    
                    # حساب إشارة موحدة للمؤشر
                    ma_trend_signal = 0
                    price_position_signal = 0
                    
                    # اتجاه المتوسط المتحرك
                    if current_ma > prev_ma:
                        ma_trend_signal = 0.6
                    elif current_ma < prev_ma:
                        ma_trend_signal = -0.6
                    
                    # موقع السعر من المتوسط
                    if current_price > current_ma:
                        price_position_signal = 0.4
                    else:
                        price_position_signal = -0.4
                    
                    # دمج الإشارتين في إشارة واحدة
                    combined_signal = ma_trend_signal + price_position_signal
                    signals.append(combined_signal)
                    
                    # إنشاء تفصيل واحد شامل
                    if combined_signal > 0.5:
                        details.append(f"✅ {ma_type.upper()} صاعد والسعر فوقه ({current_ma:.5f})")
                    elif combined_signal < -0.5:
                        details.append(f"❌ {ma_type.upper()} هابط والسعر تحته ({current_ma:.5f})")
                    elif ma_trend_signal > 0 and price_position_signal < 0:
                        details.append(f"📊 {ma_type.upper()} صاعد لكن السعر تحته ({current_ma:.5f})")
                    elif ma_trend_signal < 0 and price_position_signal > 0:
                        details.append(f"📊 {ma_type.upper()} هابط لكن السعر فوقه ({current_ma:.5f})")
                    else:
                        details.append(f"📊 {ma_type.upper()} محايد ({current_ma:.5f})")
                else:
                    print(f"لا يمكن استخراج قيم {ma_type}")
        
        return signals, details

    def analyze_momentum_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الزخم"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل RSI
        if 'rsi' in indicators_data:
            rsi_data = indicators_data['rsi']
            rsi_values = self._get_indicator_values(rsi_data, ['value', 'rsi'])
            
            if rsi_values and len(rsi_values) >= 2:
                try:
                    current_rsi = float(rsi_values[0])
                    prev_rsi = float(rsi_values[1])
                except (ValueError, TypeError) as e:
                    print(f"⚠️ خطأ في قراءة قيم RSI: {e}")
                    return signals, details
                
                # تشبع شراء/بيع
                if current_rsi < 30:
                    if current_rsi < 20:
                        signals.append(1)
                        details.append(f"✅ RSI={current_rsi:.1f} (تشبع بيع شديد)")
                    else:
                        signals.append(0.7)
                        details.append(f"✅ RSI={current_rsi:.1f} (تشبع بيع)")
                elif current_rsi > 70:
                    if current_rsi > 80:
                        signals.append(-1)
                        details.append(f"❌ RSI={current_rsi:.1f} (تشبع شراء شديد)")
                    else:
                        signals.append(-0.7)
                        details.append(f"❌ RSI={current_rsi:.1f} (تشبع شراء)")
                elif current_rsi > 50 and prev_rsi <= 50:
                    signals.append(0.6)
                    details.append(f"📈 RSI={current_rsi:.1f} (زخم صعودي)")
                elif current_rsi < 50 and prev_rsi >= 50:
                    signals.append(-0.6)
                    details.append(f"📉 RSI={current_rsi:.1f} (زخم هبوطي)")
            else:
                print("لا يمكن استخراج قيم RSI")
        
        # تحليل MACD
        if 'macd' in indicators_data:
            macd_data = indicators_data['macd']
            
            # طباعة بيانات MACD للتحقق
            if len(macd_data) > 0:
                print(f"🔍 بيانات MACD: {macd_data[0]}")
            
            macd_values = self._get_indicator_values(macd_data, ['value', 'macd', 'macd_value'])
            
            if macd_values and len(macd_values) >= 2:
                try:
                    current_macd = float(macd_values[0])
                    prev_macd = float(macd_values[1])
                except (ValueError, TypeError) as e:
                    print(f"⚠️ خطأ في قراءة قيم MACD: {e}")
                    return signals, details
                
                # استخراج خط الإشارة الحالي
                current_signal = self._get_indicator_value(macd_data, ['signal', 'macd_signal'])
                
                # استخراج قيمة الإشارة السابقة بشكل صحيح
                prev_signal = None
                if len(macd_data) > 1:
                    try:
                        prev_signal = float(macd_data[1].get('signal', macd_data[1].get('macd_signal', None)))
                    except:
                        prev_signal = None
                
                # استخراج histogram إن وُجد
                current_hist = self._get_indicator_value(macd_data, ['hist', 'macd_hist', 'histogram'])
                
                # طباعة القيم للتحقق
                print(f"📊 MACD: current={current_macd:.6f}, prev={prev_macd:.6f}")
                print(f"📊 Signal: current={current_signal}, prev={prev_signal}")
                print(f"📊 Histogram: {current_hist}")
                
                # التحليل بناءً على التقاطعات والموقع
                if current_signal is not None and prev_signal is not None:
                    # تقاطع صعودي (Bullish Crossover)
                    if current_macd > current_signal and prev_macd <= prev_signal:
                        signals.append(1.0)
                        details.append(f"✅ MACD عبر فوق خط الإشارة ({current_macd:.6f} > {current_signal:.6f})")
                    
                    # تقاطع هبوطي (Bearish Crossover)
                    elif current_macd < current_signal and prev_macd >= prev_signal:
                        signals.append(-1.0)
                        details.append(f"❌ MACD عبر تحت خط الإشارة ({current_macd:.6f} < {current_signal:.6f})")
                    
                    # MACD فوق خط الإشارة (إيجابي)
                    elif current_macd > current_signal:
                        # استخدام histogram لتحديد القوة
                        if current_hist and current_hist > 0:
                            signals.append(0.6)
                            details.append(f"📈 MACD إيجابي وقوي (hist={current_hist:.6f})")
                        else:
                            signals.append(0.4)
                            details.append(f"📈 MACD إيجابي ({current_macd:.6f} > {current_signal:.6f})")
                    
                    # MACD تحت خط الإشارة (سلبي)
                    else:
                        # استخدام histogram لتحديد القوة
                        if current_hist and current_hist < 0:
                            signals.append(-0.6)
                            details.append(f"📉 MACD سلبي وقوي (hist={current_hist:.6f})")
                        else:
                            signals.append(-0.4)
                            details.append(f"📉 MACD سلبي ({current_macd:.6f} < {current_signal:.6f})")
                
                # إذا لم يكن هناك خط إشارة، استخدم موقع MACD من الصفر
                elif current_macd > 0 and prev_macd <= 0:
                    signals.append(0.8)
                    details.append(f"✅ MACD عبر فوق الصفر ({current_macd:.6f})")
                elif current_macd < 0 and prev_macd >= 0:
                    signals.append(-0.8)
                    details.append(f"❌ MACD عبر تحت الصفر ({current_macd:.6f})")
                elif current_macd > 0:
                    signals.append(0.3)
                    details.append(f"📊 MACD فوق الصفر ({current_macd:.6f})")
                else:
                    signals.append(-0.3)
                    details.append(f"📊 MACD تحت الصفر ({current_macd:.6f})")
            else:
                print("⚠️ لا يمكن استخراج قيم MACD كافية")
        
        # تحليل Stochastic
        if 'stoch' in indicators_data:
            stoch_data = indicators_data['stoch']
            if len(stoch_data) >= 1 and ('k' in stoch_data[0] or 'value' in stoch_data[0]):
                current_k = float(stoch_data[0].get('k', stoch_data[0].get('value', 0)))
                current_d = float(stoch_data[0].get('d', 0))
                
                if current_k < 20:
                    signals.append(0.7)
                    details.append(f"✅ Stochastic={current_k:.1f} (تشبع بيع)")
                elif current_k > 80:
                    signals.append(-0.7)
                    details.append(f"❌ Stochastic={current_k:.1f} (تشبع شراء)")
        
        return signals, details

        
    def analyze_volatility_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات التذبذب"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Bollinger Bands
        if 'bbands' in indicators_data:
            bb_data = indicators_data['bbands']
            
            # طباعة بيانات Bollinger Bands للتحقق
            if len(bb_data) > 0:
                print(f"🔍 بيانات Bollinger Bands: {bb_data[0]}")
            
            if len(bb_data) >= 1 and ('upper_band' in bb_data[0] or 'value' in bb_data[0]):
                try:
                    current_price = float(price_data[-1]['close'])
                    upper_band = float(bb_data[0].get('upper_band', bb_data[0].get('value', 0)))
                    middle_band = float(bb_data[0].get('middle_band', 0))
                    lower_band = float(bb_data[0].get('lower_band', 0))
                    
                    # التحقق من صحة القيم
                    if upper_band == 0 or lower_band == 0 or middle_band == 0:
                        print(f"⚠️ قيم Bollinger Bands غير صحيحة: upper={upper_band}, middle={middle_band}, lower={lower_band}")
                        return signals, details
                    
                except (ValueError, TypeError, KeyError) as e:
                    print(f"⚠️ خطأ في قراءة بيانات Bollinger Bands: {e}")
                    return signals, details
                
                # حساب عرض النطاق (Band Width) للتحقق من التقلبات
                band_width = (upper_band - lower_band) / middle_band * 100
                
                # حساب موقع السعر النسبي (%B)
                # %B = (Price - Lower Band) / (Upper Band - Lower Band)
                percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
                
                # طباعة القيم للتحقق
                print(f"📊 Bollinger Bands:")
                print(f"   السعر: {current_price:.5f}")
                print(f"   Upper: {upper_band:.5f}, Middle: {middle_band:.5f}, Lower: {lower_band:.5f}")
                print(f"   Band Width: {band_width:.2f}%")
                print(f"   %B: {percent_b:.3f}")
                
                # تحليل موقع السعر من البولينجر
                # السعر خارج الحد العلوي (تشبع شرائي)
                if current_price >= upper_band:
                    if percent_b > 1.02:  # خارج بـ 2% أو أكثر
                        signals.append(-0.9)
                        details.append(f"❌ السعر خارج الحد العلوي - تشبع شرائي قوي (%B={percent_b:.2f})")
                    else:
                        signals.append(-0.7)
                        details.append(f"❌ السعر عند الحد العلوي - احتمال ارتداد (%B={percent_b:.2f})")
                
                # السعر خارج الحد السفلي (تشبع بيعي)
                elif current_price <= lower_band:
                    if percent_b < -0.02:  # خارج بـ 2% أو أكثر
                        signals.append(0.9)
                        details.append(f"✅ السعر خارج الحد السفلي - تشبع بيعي قوي (%B={percent_b:.2f})")
                    else:
                        signals.append(0.7)
                        details.append(f"✅ السعر عند الحد السفلي - احتمال ارتداد (%B={percent_b:.2f})")
                
                # السعر في المنطقة العليا (بين الوسط والحد العلوي)
                elif current_price > middle_band:
                    distance_from_middle = (current_price - middle_band) / (upper_band - middle_band)
                    if distance_from_middle > 0.7:  # قريب جداً من الحد العلوي
                        signals.append(-0.5)
                        details.append(f"📊 السعر قريب من الحد العلوي - حذر (%B={percent_b:.2f})")
                    else:
                        signals.append(0.2)
                        details.append(f"📊 السعر في المنطقة العليا (%B={percent_b:.2f})")
                
                # السعر في المنطقة السفلى (بين الوسط والحد السفلي)
                else:
                    distance_from_middle = (middle_band - current_price) / (middle_band - lower_band)
                    if distance_from_middle > 0.7:  # قريب جداً من الحد السفلي
                        signals.append(0.5)
                        details.append(f"📊 السعر قريب من الحد السفلي - فرصة (%B={percent_b:.2f})")
                    else:
                        signals.append(-0.2)
                        details.append(f"📊 السعر في المنطقة السفلى (%B={percent_b:.2f})")
                
                # تحذير من Bollinger Squeeze (تقلبات منخفضة)
                if band_width < 1.5:  # النطاق ضيق جداً
                    details.append(f"⚠️ Bollinger Squeeze - توقع حركة قوية قريباً (عرض={band_width:.2f}%)")
            else:
                print("⚠️ لا يمكن استخراج بيانات Bollinger Bands")
        
        # تحليل ATR
        if 'atr' in indicators_data:
            atr_data = indicators_data['atr']
            if len(atr_data) >= 2 and 'value' in atr_data[0] and 'value' in atr_data[1]:
                current_atr = float(atr_data[0]['value'])
                prev_atr = float(atr_data[1]['value'])
                atr_change = ((current_atr - prev_atr) / prev_atr * 100) if prev_atr != 0 else 0
                
                if atr_change > 10:
                    details.append(f"⚠️ ATR={current_atr:.5f} (تقلب متزايد +{atr_change:.1f}%)")
                elif atr_change < -10:
                    details.append(f"📊 ATR={current_atr:.5f} (تقلب متناقص {atr_change:.1f}%)")
                else:
                    details.append(f"📊 ATR={current_atr:.5f} (تقلب مستقر)")
        
        return signals, details

    def analyze_volume_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الحجم"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Money Flow Index
        if 'mfi' in indicators_data:
            mfi_data = indicators_data['mfi']
            if len(mfi_data) >= 1 and 'value' in mfi_data[0]:
                current_mfi = float(mfi_data[0]['value'])
                
                if current_mfi < 20:
                    signals.append(0.7)
                    details.append(f"✅ MFI={current_mfi:.1f} (تشبع بيع)")
                elif current_mfi > 80:
                    signals.append(-0.7)
                    details.append(f"❌ MFI={current_mfi:.1f} (تشبع شراء)")
                elif current_mfi > 50:
                    signals.append(0.3)
                    details.append(f"📈 MFI={current_mfi:.1f} (زخم إيجابي)")
                else:
                    signals.append(-0.3)
                    details.append(f"📉 MFI={current_mfi:.1f} (زخم سلبي)")
        
        # تحليل Chaikin Money Flow
        if 'cmf' in indicators_data:
            cmf_data = indicators_data['cmf']
            if len(cmf_data) >= 1 and 'value' in cmf_data[0]:
                current_cmf = float(cmf_data[0]['value'])
                
                if current_cmf > 0.1:
                    signals.append(0.5)
                    details.append(f"📈 CMF={current_cmf:.3f} (تدفق أموال إيجابي)")
                elif current_cmf < -0.1:
                    signals.append(-0.5)
                    details.append(f"📉 CMF={current_cmf:.3f} (تدفق أموال سلبي)")
        
        return signals, details

    def analyze_price_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات السعر"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Average Price
        if 'avgprice' in indicators_data:
            avgprice_data = indicators_data['avgprice']
            if len(avgprice_data) >= 1:
                current_avg = float(avgprice_data[0]['value'])
                current_close = float(price_data[-1]['close'])
                
                if current_close > current_avg:
                    signals.append(0.4)
                    details.append(f"📈 السعر فوق المتوسط")
                else:
                    signals.append(-0.4)
                    details.append(f"📉 السعر تحت المتوسط")
        
        return signals, details

    def analyze_misc_indicators(self, indicators_data, price_data):
        """تحليل المؤشرات الأخرى"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Parabolic SAR
        if 'sar' in indicators_data:
            sar_data = indicators_data['sar']
            if len(sar_data) >= 1:
                current_sar = float(sar_data[0]['value'])
                current_close = float(price_data[-1]['close'])
                
                if current_close > current_sar:
                    signals.append(0.6)
                    details.append(f"📈 SAR={current_sar:.5f} (إشارة صعود)")
                else:
                    signals.append(-0.6)
                    details.append(f"📉 SAR={current_sar:.5f} (إشارة هبوط)")
        
        # تحليل Ultimate Oscillator
        if 'ultosc' in indicators_data:
            ultosc_data = indicators_data['ultosc']
            if len(ultosc_data) >= 1:
                current_ultosc = float(ultosc_data[0]['value'])
                
                if current_ultosc < 30:
                    signals.append(0.7)
                    details.append(f"✅ Ultimate={current_ultosc:.1f} (تشبع بيع)")
                elif current_ultosc > 70:
                    signals.append(-0.7)
                    details.append(f"❌ Ultimate={current_ultosc:.1f} (تشبع شراء)")
        
        return signals, details
    #--------------------------------------------------
    def train_ai_models(self, pair, historical_data):
        """تدريب نماذج الذكاء الاصطناعي"""
        try:
            if not historical_data or len(historical_data) < 100:
                print(f"بيانات تاريخية غير كافية للتدريب لـ {pair}")
                return False
            
            # تحضير بيانات التدريب من البيانات التاريخية فقط (دون مؤشرات إضافية)
            features, labels = self.ai_system.prepare_training_data(historical_data, {})
            
            if features is None or len(features) < 50:
                print(f"بيانات تدريب غير كافية لـ {pair}")
                return False
            
            # تدريب النماذج
            print(f"🤖 بدء تدريب النماذج لـ {pair}...")
            
            # تدريب نماذج ML
            ml_success = self.ai_system.train_ml_models(features, labels)
            
            # تدريب نماذج Neural Networks
            lstm_success = self.ai_system.train_lstm_model(features, labels)
            cnn_success = self.ai_system.train_cnn_model(features, labels)
            
            # حفظ النماذج
            self.ai_system.save_models()
            
            print(f"✅ تم تدريب النماذج لـ {pair} - ML: {ml_success}, LSTM: {lstm_success}, CNN: {cnn_success}")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تدريب النماذج لـ {pair}: {e}")
            return False
    
    def train_ai_with_evaluations(self, trade_type='real'):
        """تدريب الذكاء الاصطناعي بناءً على نوع الصفقات المُقيَّمة
        trade_type: 'real' لتدريب AI الحقيقي على صفقات حقيقية، أو 'test' لتدريب تجريبي للمستخدم
        """
        try:
            # فلترة الصفقات المُقيَّمة حسب النوع المطلوب
            trade_type = 'real' if trade_type not in ['real', 'test'] else trade_type
            evaluated_trades = [trade for trade in self.trade_history
                                if trade.get('trade_type') == trade_type and
                                trade.get('user_evaluation') in ['successful', 'failed']]

            if len(evaluated_trades) < 1:
                print(f"❌ لا توجد صفقات مُقيَّمة كافية للتدريب لنوع '{trade_type}' (مطلوب صفقة واحدة على الأقل، الموجود: {len(evaluated_trades)})")
                return False
            
            # تحضير البيانات من التقييمات الحقيقية فقط
            features = []
            labels = []
            
            for trade in evaluated_trades:
                    # إعداد الميزات من بيانات المؤشرات
                    feature_vector = []

                    # إضافة بيانات الأسعار
                    indicators = trade.get('indicators_data', {})
                    feature_vector.extend([
                        indicators.get('open', 0),
                        indicators.get('high', 0),
                        indicators.get('low', 0),
                        indicators.get('close', 0)
                    ])

                    # إضافة قيم المؤشرات بحد أقصى 2 مؤشر لضمان ثبات الأبعاد
                    indicator_count = 0
                    for key, value in indicators.items():
                        if key in ['open', 'high', 'low', 'close']:
                            continue
                        if indicator_count >= 2:
                            break
                        try:
                            if value is not None:
                                feature_vector.append(float(value))
                            else:
                                feature_vector.append(0)
                        except Exception:
                            feature_vector.append(0)
                        indicator_count += 1

                    # التأكد من عدد الميزات ثابت (14)
                    if len(feature_vector) > 14:
                        feature_vector = feature_vector[:14]
                    while len(feature_vector) < 14:
                        feature_vector.append(0)
                    
                    # إعداد التسمية
                    if trade['user_evaluation'] == 'successful':
                        if trade['signal'] == 'CALL':
                            label = 1  # BUY ناجح
                        else:
                            label = 0  # SELL ناجح
                    else:
                        if trade['signal'] == 'CALL':
                            label = 0  # BUY فاشل
                        else:
                            label = 1  # SELL فاشل
                    
                    features.append(feature_vector)
                    labels.append(label)
            
            if len(features) < 10:
                print("❌ بيانات تدريب غير كافية من التقييمات")
                return False
            
            print(f"🤖 تدريب النماذج على {len(features)} تقييم صفقة ({'حقيقية' if trade_type=='real' else 'تجريبية'})...")
            
            # تدريب النماذج
            ml_success = self.ai_system.train_ml_models(np.array(features), np.array(labels))
            lstm_success = self.ai_system.train_lstm_model(np.array(features), np.array(labels))
            cnn_success = self.ai_system.train_cnn_model(np.array(features), np.array(labels))
            
            # حفظ النماذج
            self.ai_system.save_models()
            
            print(f"✅ تم تدريب النماذج على {len(features)} صفقة {'حقيقية' if trade_type=='real' else 'تجريبية'} - ML: {ml_success}, LSTM: {lstm_success}, CNN: {cnn_success}")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تدريب النماذج على التقييمات: {e}")
            return False

    def analyze_indicators_only(self, indicators_data, price_data, selected_indicators):
        """تحليل المؤشرات من API فقط - بدون أي تدخل من AI"""
        if not indicators_data or not price_data:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'بيانات غير كافية',
                'details': [],
                'indicators': {}
            }

        all_signals = []
        all_details = []

        # تحليل كل مجموعة مؤشرات من API فقط
        for category, indicators in selected_indicators.items():
            if not indicators:
                continue

            category_indicators = {ind: indicators_data.get(ind, []) for ind in indicators if ind in indicators_data}

            if category == 'trend':
                signals, details = self.analyze_trend_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'momentum':
                signals, details = self.analyze_momentum_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volatility':
                signals, details = self.analyze_volatility_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volume':
                signals, details = self.analyze_volume_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'price':
                signals, details = self.analyze_price_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            else:
                signals, details = self.analyze_misc_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)

        # حساب الإشارة النهائية من المؤشرات فقط
        traditional_avg = np.mean(all_signals) if all_signals else 0
        threshold = 0.35
        confidence = min(abs(traditional_avg) * 100, 100)
        
        if traditional_avg > threshold:
            final_signal = 'CALL'
            signal_text = 'صعود (CALL) 🟢'
        elif traditional_avg < -threshold:
            final_signal = 'PUT'
            signal_text = 'هبوط (PUT) 🔴'
        else:
            final_signal = 'HOLD'
            signal_text = 'انتظار (HOLD) ⚪'

        # جمع قيم المؤشرات
        indicators_values = {}
        for category, indicators in selected_indicators.items():
            for indicator in indicators:
                if indicator in indicators_data and indicators_data[indicator]:
                    latest_data = indicators_data[indicator][0]
                    if isinstance(latest_data, dict):
                        for key, value in latest_data.items():
                            if key != 'datetime':
                                indicators_values[f"{indicator}_{key}"] = self.safe_float(value)

        return {
            'signal': final_signal,
            'signal_text': signal_text,
            'confidence': round(confidence, 1),
            'details': all_details,
            'indicators': indicators_values,
            'all_details': all_details
        }
    
    def calculate_python_indicators(self, price_data):
        """حساب المؤشرات باستخدام مكتبات Python المحلية"""
        try:
            if len(price_data) < 20:
                return None
            
            # تحويل البيانات إلى arrays
            closes = np.array([p['close'] for p in price_data])
            highs = np.array([p['high'] for p in price_data])
            lows = np.array([p['low'] for p in price_data])
            
            indicators = {}
            signals = []
            details = []
            
            # حساب RSI يدوياً
            if len(closes) >= 14:
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                indicators['rsi'] = rsi
                
                if rsi < 30:
                    signals.append(0.8)
                    details.append(f"✅ RSI المحسوب={rsi:.1f} (تشبع بيع)")
                elif rsi > 70:
                    signals.append(-0.8)
                    details.append(f"❌ RSI المحسوب={rsi:.1f} (تشبع شراء)")
                else:
                    signals.append(0.0)
                    details.append(f"📊 RSI المحسوب={rsi:.1f} (محايد)")
            
            # حساب MACD يدوياً
            if len(closes) >= 26:
                ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
                ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                histogram = macd_line - signal_line
                
                indicators['macd'] = float(macd_line.iloc[-1])
                indicators['macd_signal'] = float(signal_line.iloc[-1])
                indicators['macd_hist'] = float(histogram.iloc[-1])
                
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    signals.append(0.6)
                    details.append(f"📈 MACD المحسوب إيجابي ({macd_line.iloc[-1]:.6f})")
                else:
                    signals.append(-0.6)
                    details.append(f"📉 MACD المحسوب سلبي ({macd_line.iloc[-1]:.6f})")
            
            # حساب Bollinger Bands يدوياً
            if len(closes) >= 20:
                sma = np.mean(closes[-20:])
                std = np.std(closes[-20:])
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                current_price = closes[-1]
                
                indicators['bb_upper'] = upper_band
                indicators['bb_middle'] = sma
                indicators['bb_lower'] = lower_band
                
                # حساب %B
                percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
                indicators['bb_percent_b'] = percent_b
                
                if current_price >= upper_band:
                    signals.append(-0.7)
                    details.append(f"❌ BB: السعر عند العلوي (%B={percent_b:.2f})")
                elif current_price <= lower_band:
                    signals.append(0.7)
                    details.append(f"✅ BB: السعر عند السفلي (%B={percent_b:.2f})")
                else:
                    signals.append(0.0)
                    details.append(f"📊 BB: السعر وسط النطاق (%B={percent_b:.2f})")
            
            # حساب ADX إذا كان TA-Lib متاحاً
            if TALIB_AVAILABLE and len(closes) >= 14:
                try:
                    adx = talib.ADX(highs, lows, closes, timeperiod=14)
                    if not np.isnan(adx[-1]):
                        indicators['adx'] = float(adx[-1])
                        if adx[-1] > 25:
                            details.append(f"📊 ADX={adx[-1]:.1f} (اتجاه قوي)")
                        else:
                            details.append(f"📊 ADX={adx[-1]:.1f} (اتجاه ضعيف)")
                except:
                    pass
            
            # حساب ATR (Average True Range)
            if len(price_data) >= 14:
                tr_list = []
                for i in range(1, len(price_data)):
                    high = price_data[i]['high']
                    low = price_data[i]['low']
                    prev_close = price_data[i-1]['close']
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    tr_list.append(tr)
                atr = np.mean(tr_list[-14:])
                indicators['atr'] = atr
                details.append(f"📊 ATR={atr:.5f} (تقلب)")
            
            # حساب التقلب (Volatility)
            if len(closes) >= 20:
                volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
                indicators['volatility'] = volatility
                details.append(f"📊 التقلب={volatility:.2f}%")
            
            # حساب الإشارة النهائية
            avg_signal = np.mean(signals) if signals else 0
            confidence = abs(avg_signal) * 100
            
            if avg_signal > 0.35:
                signal = 'CALL'
                signal_text = 'صعود (CALL) 🟢'
            elif avg_signal < -0.35:
                signal = 'PUT'
                signal_text = 'هبوط (PUT) 🔴'
            else:
                signal = 'HOLD'
                signal_text = 'انتظار (HOLD) ⚪'
            
            return {
                'signal': signal,
                'signal_text': signal_text,
                'confidence': round(min(confidence, 100), 1),
                'indicators': indicators,
                'details': details,
                'avg_signal': avg_signal
            }
            
        except Exception as e:
            print(f"❌ خطأ في حساب مؤشرات Python: {e}")
            return None
    
    def calculate_trade_duration(self, confidence, volatility=None):
        """حساب مدة الصفقة بناءً على التحليل"""
        try:
            # القاعدة الأساسية: ثقة أعلى = مدة أطول
            base_duration = 5  # 5 دقائق
            
            if confidence >= 80:
                duration = 10  # ثقة عالية جداً = 10 دقائق
            elif confidence >= 70:
                duration = 8
            elif confidence >= 60:
                duration = 6
            elif confidence >= 50:
                duration = 5
            else:
                duration = 3  # ثقة منخفضة = 3 دقائق فقط
            
            # تعديل بناءً على التقلب إن وُجد
            if volatility is not None:
                if volatility > 2.0:  # تقلب عالي
                    duration = max(3, duration - 2)  # تقليل المدة
                elif volatility < 0.5:  # تقلب منخفض
                    duration = min(15, duration + 2)  # زيادة المدة
            
            return duration
            
        except Exception as e:
            print(f"❌ خطأ في حساب مدة الصفقة: {e}")
            return 5  # مدة افتراضية
    
    def generate_signal(self, indicators_data, price_data, selected_indicators):
        """توليد إشارة التداول مع الذكاء الاصطناعي"""
        if not indicators_data or not price_data:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'بيانات غير كافية',
                'indicators': {},
                'ai_prediction': None
            }

        all_signals = []
        all_details = []

        # تحليل كل مجموعة مؤشرات
        for category, indicators in selected_indicators.items():
            if not indicators:  # تخطي المجموعات الفارغة
                continue

            category_indicators = {ind: indicators_data.get(ind, []) for ind in indicators if ind in indicators_data}

            if category == 'trend':
                signals, details = self.analyze_trend_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'momentum':
                signals, details = self.analyze_momentum_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volatility':
                signals, details = self.analyze_volatility_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volume':
                signals, details = self.analyze_volume_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'price':
                signals, details = self.analyze_price_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'misc':
                signals, details = self.analyze_misc_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)

        # تحليل الذكاء الاصطناعي
        ai_prediction = None
        if self.ai_enabled and (self.ai_system.ml_models or self.ai_system.neural_models):
            try:
                # إعداد بيانات للتنبؤ
                current_features = []
                current_price = price_data[-1]
                current_features.extend([
                    current_price['open'],
                    current_price['high'],
                    current_price['low'],
                    current_price['close']
                ])

                # إكمال الميزات لتصل 11 (4 أسعار + 7 ميزات فنية كحشو إن لم تتوفر)
                while len(current_features) < 11:
                    current_features.append(0)

                # إضافة قيم المؤشرات (حد أقصى 3 مؤشرات مثل التحضير الأصلي)
                indicator_count = 0
                max_indicators = 3
                for category, indicators in selected_indicators.items():
                    for indicator in indicators:
                        if indicator_count >= max_indicators:
                            break
                        if indicator in indicators_data and indicators_data[indicator]:
                            indicator_value = self.ai_system._extract_indicator_value(indicators_data[indicator][0])
                            current_features.append(indicator_value if indicator_value is not None else 0)
                        else:
                            current_features.append(0)
                        indicator_count += 1

                # التأكد من عدد الميزات ثابت (14)
                while len(current_features) < 14:
                    current_features.append(0)
                
                # الحصول على تنبؤ الذكاء الاصطناعي
                ai_prediction = self.ai_system.predict_with_ensemble(current_features)
                
                # تحليل الأنماط في الرسم البياني
                chart_analysis = self.ai_system.analyze_chart_patterns(price_data)
                ai_prediction['chart_patterns'] = chart_analysis
                
                # تنبؤ التعلم المعزز
                state_vector = current_features[:10]  # أول 10 قيم كحالة
                rl_prediction = self.ai_system.get_reinforcement_prediction(state_vector)
                ai_prediction['reinforcement_learning'] = rl_prediction
                
            except Exception as e:
                print(f"❌ خطأ في تنبؤ الذكاء الاصطناعي: {e}")
                ai_prediction = {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                    'ensemble_used': False
                }

        # دمج النتائج التقليدية مع الذكاء الاصطناعي
        traditional_avg = np.mean(all_signals) if all_signals else 0
        threshold = 0.35
        
        # طباعة تفصيلية للتحليل (للتصحيح)
        print(f"🔍 تحليل الإشارات:")
        print(f"   📊 عدد الإشارات: {len(all_signals)}")
        print(f"   📈 الإشارات الإيجابية: {len([s for s in all_signals if s > 0])} ({sum([s for s in all_signals if s > 0]):.2f})")
        print(f"   📉 الإشارات السلبية: {len([s for s in all_signals if s < 0])} ({sum([s for s in all_signals if s < 0]):.2f})")
        print(f"   ⚖️ المتوسط: {traditional_avg:.3f}")
        print(f"   📋 أول 5 تفاصيل: {all_details[:5]}")
        
        # إذا كان لدينا تنبؤ من الذكاء الاصطناعي، ندمجه مع النتائج التقليدية
        if ai_prediction and ai_prediction.get('ensemble_used', False):
            ai_signal = ai_prediction['signal']
            ai_confidence = ai_prediction['confidence']
            
            # دمج الإشارات (وزن 70% للذكاء الاصطناعي، 30% للتحليل التقليدي)
            if ai_signal == 'BUY':
                ai_weight = 0.7
                traditional_weight = 0.3
                final_signal = 'CALL'
                signal_text = f'صعود (CALL) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence * ai_weight + abs(traditional_avg) * 100 * traditional_weight
            elif ai_signal == 'SELL':
                ai_weight = 0.7
                traditional_weight = 0.3
                final_signal = 'PUT'
                signal_text = f'هبوط (PUT) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence * ai_weight + abs(traditional_avg) * 100 * traditional_weight
            else:
                final_signal = 'HOLD'
                signal_text = f'انتظار (HOLD) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence
        else:
            # استخدام التحليل التقليدي فقط
            confidence = min(abs(traditional_avg) * 100, 100)
            
            if traditional_avg > threshold:
                final_signal = 'CALL'
                signal_text = 'صعود (CALL) 🟢'
            elif traditional_avg < -threshold:
                final_signal = 'PUT'
                signal_text = 'هبوط (PUT) 🔴'
            else:
                final_signal = 'HOLD'
                signal_text = 'انتظار (HOLD) ⚪'

        # جمع قيم المؤشرات
        indicators_values = {}
        for category, indicators in selected_indicators.items():
            for indicator in indicators:
                if indicator in indicators_data and indicators_data[indicator]:
                    latest_data = indicators_data[indicator][0]
                    if isinstance(latest_data, dict):
                        for key, value in latest_data.items():
                            if key != 'datetime':
                                indicators_values[f"{indicator}_{key}"] = self.safe_float(value)

        # إضافة تفاصيل الذكاء الاصطناعي
        ai_details = []
        if ai_prediction:
            if ai_prediction.get('probabilities'):
                ai_details.append(f"🤖 AI احتمالات: BUY {ai_prediction['probabilities'].get('BUY', 0):.1f}%, SELL {ai_prediction['probabilities'].get('SELL', 0):.1f}%, HOLD {ai_prediction['probabilities'].get('HOLD', 0):.1f}%")
            
            if ai_prediction.get('chart_patterns', {}).get('patterns_detected'):
                patterns = ai_prediction['chart_patterns']['patterns_detected']
                ai_details.append(f"📊 أنماط الرسم: {', '.join(patterns)}")
            
            if ai_prediction.get('reinforcement_learning', {}).get('signal'):
                rl_signal = ai_prediction['reinforcement_learning']['signal']
                rl_conf = ai_prediction['reinforcement_learning']['confidence']
                ai_details.append(f"🧠 التعلم المعزز: {rl_signal} ({rl_conf:.1f}%)")

        # دمج التفاصيل
        all_details.extend(ai_details)

        return {
            'signal': final_signal,
            'signal_text': signal_text,
            'confidence': round(confidence, 1),
            'reason': ' | '.join(all_details[:4]) if all_details else 'لا توجد إشارات',
            'all_details': all_details,
            'indicators': indicators_values,
            'price': {
                'open': self.safe_float(price_data[-1]['open']),
                'high': self.safe_float(price_data[-1]['high']),
                'low': self.safe_float(price_data[-1]['low']),
                'close': self.safe_float(price_data[-1]['close'])
            },
            'last_candle_time': price_data[-1]['datetime'],
            'ai_prediction': ai_prediction,
            'ai_enabled': self.ai_enabled
        }
    def generate_three_analyses(self, indicators_data, price_data, selected_indicators, trade_duration=5):
        """توليد 3 نتائج تحليل منفصلة: API, Python, RL
        
        Args:
            trade_duration: مدة الصفقة بالدقائق (من اختيار المستخدم)
        """
        current_time = datetime.now()
        
        # حساب وقت الدخول: بداية الدقيقة التالية (الثواني = 00)
        next_minute = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
        
        # حساب وقت الخروج: بداية الدقيقة بعد انتهاء المدة (الثواني = 00)
        exit_time = (next_minute + timedelta(minutes=trade_duration)).replace(second=0, microsecond=0)
        
        # 1️⃣ النتيجة الأولى: تحليل المؤشرات من API فقط (بدون AI)
        api_analysis = self.analyze_indicators_only(indicators_data, price_data, selected_indicators)
        api_confidence = api_analysis.get('confidence', 50)
        
        api_result = {
            'type': 'api',
            'title': 'تحليل المؤشرات (API)',
            'icon': '📊',
            'signal': api_analysis['signal'],
            'signal_text': api_analysis['signal_text'],
            'confidence': api_confidence,
            'details': api_analysis.get('details', [])[:5],
            'entry_time': next_minute.strftime('%H:%M:%S'),
            'exit_time': exit_time.strftime('%H:%M:%S'),
            'duration': trade_duration,
            'analysis_time': current_time.strftime('%H:%M:%S')
        }
        
        # 2️⃣ النتيجة الثانية: تحليل مكتبات Python
        python_result = None
        python_analysis = self.calculate_python_indicators(price_data)
        if python_analysis:
            py_confidence = python_analysis['confidence']
            
            python_result = {
                'type': 'python',
                'title': 'تحليل مكتبات Python',
                'icon': '🐍',
                'signal': python_analysis['signal'],
                'signal_text': python_analysis['signal_text'],
                'confidence': py_confidence,
                'details': python_analysis['details'][:5],
                'entry_time': next_minute.strftime('%H:%M:%S'),
                'exit_time': exit_time.strftime('%H:%M:%S'),
                'duration': trade_duration,
                'analysis_time': current_time.strftime('%H:%M:%S')
            }
        else:
            # في حالة فشل حساب Python، استخدم قيم افتراضية
            python_result = {
                'type': 'python',
                'title': 'تحليل مكتبات Python',
                'icon': '🐍',
                'signal': 'HOLD',
                'signal_text': 'انتظار (HOLD) ⚪',
                'confidence': 0,
                'details': ['⚠️ بيانات غير كافية للتحليل'],
                'entry_time': next_minute.strftime('%H:%M:%S'),
                'exit_time': exit_time.strftime('%H:%M:%S'),
                'duration': trade_duration,
                'analysis_time': current_time.strftime('%H:%M:%S')
            }
        
        # 3️⃣ النتيجة الثالثة: التعلم المعزز (Q-Learning)
        # التأكد من وجود نظام AI
        rl_signal = 'HOLD'
        rl_confidence = 0
        rl_details = []
        is_trained = False
        
        if hasattr(self, 'ai_system'):
            try:
                # إعداد state vector للتعلم المعزز
                state_vector = []
                current_price = price_data[-1]
                state_vector.extend([
                    current_price['open'],
                    current_price['high'],
                    current_price['low'],
                    current_price['close']
                ])
                
                # إضافة بعض المؤشرات إن وُجدت
                if python_analysis and python_analysis['indicators']:
                    ind = python_analysis['indicators']
                    state_vector.extend([
                        ind.get('rsi', 50),
                        ind.get('macd', 0),
                        ind.get('bb_percent_b', 0.5),
                        ind.get('volatility', 1.0)
                    ])
                
                # الحد من الطول إلى 10
                state_vector = state_vector[:10]
                while len(state_vector) < 10:
                    state_vector.append(0)
                
                # الحصول على تنبؤ RL
                rl_prediction = self.ai_system.get_reinforcement_prediction(state_vector)
                
                if rl_prediction and rl_prediction.get('signal'):
                    rl_signal = rl_prediction['signal']
                    rl_confidence = min(rl_prediction.get('confidence', 0), 100)
                    
                    # التحقق من وجود Q-Table مدربة
                    q_values = rl_prediction.get('q_values', {})
                    is_trained = any(abs(v) > 0.1 for v in q_values.values())
                    
                    if is_trained:
                        rl_details.append(f"🎯 Q-Values:")
                        rl_details.append(f"  • BUY: {q_values.get('BUY', 0):.2f}")
                        rl_details.append(f"  • SELL: {q_values.get('SELL', 0):.2f}")
                        rl_details.append(f"  • HOLD: {q_values.get('HOLD', 0):.2f}")
                        
                        # عدد الصفقات المدربة (تقريبي من حجم Q-table)
                        num_states = len(self.ai_system.q_table)
                        rl_details.append(f"📚 حالات مدربة: {num_states}")
                    else:
                        rl_details.append("⚠️ النموذج غير مدرب بعد")
                        rl_details.append("💡 قم بتقييم بعض الصفقات لتدريب النموذج")
            
            except Exception as e:
                print(f"❌ خطأ في تنبؤ RL: {e}")
                rl_details.append("❌ خطأ في التنبؤ")
        else:
            rl_details.append("⚠️ نظام AI غير متاح")
        
        # استخدام نفس المدة لجميع النتائج
        rl_result = {
            'type': 'reinforcement',
            'title': 'التعلم المعزز (Q-Learning)',
            'icon': '🤖',
            'signal': rl_signal,
            'signal_text': f"{'صعود (CALL) 🟢' if rl_signal == 'BUY' or rl_signal == 'CALL' else 'هبوط (PUT) 🔴' if rl_signal == 'SELL' or rl_signal == 'PUT' else 'انتظار (HOLD) ⚪'}",
            'confidence': round(rl_confidence, 1),
            'details': rl_details if rl_details else ['⚠️ لم يتم التدريب بعد'],
            'entry_time': next_minute.strftime('%H:%M:%S'),
            'exit_time': exit_time.strftime('%H:%M:%S'),
            'duration': trade_duration,
            'analysis_time': current_time.strftime('%H:%M:%S'),
            'is_trained': is_trained
        }
        
        return [api_result, python_result, rl_result]
    
    def analyze_pair(self, pair, period, selected_indicators, interval='1min', trade_duration=5):
        """تحليل زوج واحد - يُرجع 3 نتائج منفصلة
        
        Args:
            trade_duration: مدة الصفقة بالدقائق (من اختيار المستخدم)
        """
        try:
            # جلب بيانات الأسعار
            price_df = self.fetch_price_data(pair, interval, period)
            if price_df is None or len(price_df) == 0:
                print(f"لا توجد بيانات أسعار لـ {pair}")
                return None

            # جلب بيانات المؤشرات
            indicators_data = self.fetch_indicators_data(pair, selected_indicators, interval)
            if not indicators_data:
                print(f"لا توجد بيانات مؤشرات لـ {pair}")
                return None

            # تحويل بيانات الأسعار إلى تنسيق مناسب
            price_data = []
            for _, row in price_df.tail(50).iterrows():
                price_data.append({
                    'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'open': self.safe_float(row['open']),
                    'high': self.safe_float(row['high']),
                    'low': self.safe_float(row['low']),
                    'close': self.safe_float(row['close'])
                })

            if not price_data:
                print(f"بيانات الأسعار فارغة لـ {pair}")
                return None

            # توليد 3 نتائج منفصلة (مع المدة المختارة من المستخدم)
            analyses = self.generate_three_analyses(indicators_data, price_data, selected_indicators, trade_duration)
            
            # إضافة بيانات الشموع للرسم البياني
            chart_data = []
            for row in price_data:
                chart_data.append({
                    'time': row['datetime'].split(' ')[1][:5],  # الوقت فقط
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
            
            # إرجاع النتائج الثلاثة مع البيانات الإضافية
            return {
                'analyses': analyses,  # القائمة بـ 3 نتائج
                'chart_data': chart_data,
                'pair': pair,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"خطأ في تحليل {pair}: {str(e)}")
            return None
    def start_analysis(self, pairs, period, interval_minutes, selected_indicators):
        """تخزين الإعدادات دون تشغيل تحليل مستمر"""
        self.latest_results = {}  # إعادة تعيين النتائج
        self.is_running = True
        self.analysis_config = {
            'pairs': pairs,
            'period': period,
            'interval': interval_minutes,
            'indicators': selected_indicators
        }
        return True


    def stop_analysis(self):
        """إيقاف التحليل"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2)
        return True

load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")
analyzer = TradingAnalyzer(API_KEY)

app = Flask(__name__)

CORS(app)

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/pairs')
def get_pairs():
    """الحصول على قائمة الأزواج المتاحة"""
    return jsonify(analyzer.available_pairs)

@app.route('/api/indicators')
def get_indicators():
    """الحصول على قائمة المؤشرات المتاحة"""
    return jsonify(analyzer.available_indicators)

@app.route('/api/config', methods=['POST'])
def set_config():
    """تعيين إعدادات API"""
    data = request.json
    api_key = data.get('api_key', '')
    
    # إذا كان المفتاح فارغاً، استخدم المفتاح الافتراضي من .env
    if not api_key:
        api_key = API_KEY
    
    analyzer.set_api_key(api_key)
    print(f"🔑 تم تحديث مفتاح API: {'مخصص' if data.get('api_key', '') else 'افتراضي'}")
    return jsonify({'status': 'success', 'message': 'تم تحديث مفتاح API بنجاح'})

@app.route('/api/update-api-key', methods=['POST'])
def update_api_key():
    """تحديث مفتاح API فوراً"""
    try:
        data = request.json
        api_key = data.get('api_key', '')
        
        # إذا كان المفتاح فارغاً، استخدم المفتاح الافتراضي من .env
        if not api_key:
            api_key = API_KEY
            key_type = 'افتراضي'
        else:
            key_type = 'مخصص'
        
        analyzer.set_api_key(api_key)
        print(f"🔑 تم تحديث مفتاح API إلى: {key_type}")
        
        return jsonify({
            'status': 'success', 
            'message': f'تم تحديث مفتاح API إلى {key_type}',
            'key_type': key_type
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تحديث مفتاح API: {str(e)}'})

@app.route('/api/api-key-status')
def get_api_key_status():
    """الحصول على حالة مفتاح API الحالي"""
    try:
        current_key = analyzer.api_key
        is_default = current_key == API_KEY
        
        return jsonify({
            'status': 'success',
            'current_key_type': 'افتراضي' if is_default else 'مخصص',
            'is_default': is_default,
            'has_key': bool(current_key)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على حالة مفتاح API: {str(e)}'})

@app.route('/api/start', methods=['POST'])
def start_analysis():
    """بدء التحليل"""
    data = request.json
    pairs = data.get('pairs', [])
    period = data.get('period', 250)
    interval = data.get('interval', 1)
    selected_indicators = data.get('indicators', ['RSI', 'MACD'])
    api_key = data.get('api_key', '')
    
    # تحديث مفتاح API إذا تم توفيره
    if api_key:
        analyzer.set_api_key(api_key)
        print(f"🔑 تم تحديث مفتاح API من واجهة المستخدم")
    elif not analyzer.api_key:
        # إذا لم يكن هناك مفتاح API محدد، استخدم الافتراضي
        analyzer.set_api_key(API_KEY)
        print(f"🔑 استخدام المفتاح الافتراضي من .env")
    
    if analyzer.start_analysis(pairs, period, interval, selected_indicators):
        return jsonify({'status': 'success', 'message': 'تم بدء التحليل'})
    else:
        return jsonify({'status': 'error', 'message': 'التحليل يعمل بالفعل'})

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    """إيقاف التحليل"""
    if analyzer.stop_analysis():
        return jsonify({'status': 'success', 'message': 'تم إيقاف التحليل'})
    else:
        return jsonify({'status': 'error', 'message': 'فشل إيقاف التحليل'})

@app.route('/api/status')
def get_status():
    """الحصول على حالة التحليل"""
    return jsonify({
        'is_running': analyzer.is_running,
        'results': analyzer.latest_results
    })

@app.route('/api/requests-status')
def get_requests_status():
    """الحصول على حالة طلبات API"""
    return jsonify(analyzer.get_api_status())

@app.route('/api/ai/train', methods=['POST'])
def train_ai_models():
    """تدريب نماذج الذكاء الاصطناعي"""
    try:
        data = request.get_json(silent=True) or {}
        pair = data.get('pair', 'EUR/USD')
        period = data.get('period', 500)
        
        # جلب بيانات تاريخية للتدريب
        price_df = analyzer.fetch_price_data(pair, '1min', period)
        if price_df is None:
            return jsonify({'status': 'error', 'message': 'فشل في جلب البيانات التاريخية'})
        
        # تحويل البيانات
        historical_data = []
        for _, row in price_df.iterrows():
            historical_data.append({
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
        
        # تدريب النماذج
        success = analyzer.train_ai_models(pair, historical_data)
        
        if success:
            return jsonify({
                'status': 'success', 
                'message': f'تم تدريب النماذج بنجاح لـ {pair}',
                'metrics': analyzer.ai_system.get_performance_metrics()
            })
        else:
            return jsonify({'status': 'error', 'message': 'فشل في تدريب النماذج'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في التدريب: {str(e)}'})

@app.route('/api/ai/status')
def get_ai_status():
    """الحصول على حالة نظام الذكاء الاصطناعي"""
    try:
        metrics = analyzer.ai_system.get_performance_metrics()
        return jsonify({
            'status': 'success',
            'ai_enabled': analyzer.ai_enabled,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على حالة AI: {str(e)}'})

@app.route('/api/ai/toggle', methods=['POST'])
def toggle_ai():
    """تفعيل/إلغاء تفعيل الذكاء الاصطناعي"""
    try:
        data = request.get_json(silent=True) or {}
        enabled = data.get('enabled', True)
        analyzer.ai_enabled = enabled
        
        return jsonify({
            'status': 'success',
            'message': f'تم {"تفعيل" if enabled else "إلغاء تفعيل"} الذكاء الاصطناعي',
            'ai_enabled': analyzer.ai_enabled
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تغيير حالة AI: {str(e)}'})

@app.route('/api/ai/performance')
def get_ai_performance():
    """الحصول على أداء النماذج"""
    try:
        performance = analyzer.ai_system.get_performance_metrics()
        return jsonify({
            'status': 'success',
            'performance': performance
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على الأداء: {str(e)}'})

@app.route('/api/system/info')
def get_system_info():
    """معلومات النظام والمكتبات المستخدمة"""
    try:
        return jsonify({
            'status': 'success',
            'talib_available': TALIB_AVAILABLE
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على معلومات النظام: {str(e)}'})

# API endpoints لتقييم الصفقات
@app.route('/api/trades/pending')
def get_pending_trades():
    """الحصول على الصفقات المعلقة للتقييم"""
    try:
        pending_trades = analyzer.get_pending_evaluations()
        return jsonify({
            'status': 'success',
            'trades': pending_trades
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على الصفقات: {str(e)}'})

@app.route('/api/trades/history')
def get_trade_history():
    """الحصول على تاريخ الصفقات"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = analyzer.get_trade_history(limit)
        return jsonify({
            'status': 'success',
            'trades': history
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على التاريخ: {str(e)}'})

@app.route('/api/trades/statistics')
def get_trade_statistics():
    """الحصول على إحصائيات الصفقات"""
    try:
        stats = analyzer.get_trade_statistics()
        return jsonify({
            'status': 'success',
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على الإحصائيات: {str(e)}'})

@app.route('/api/trades/evaluate', methods=['POST'])
def evaluate_trade():
    """تقييم صفقة"""
    try:
        data = request.get_json(silent=True) or {}
        trade_id = data.get('trade_id')
        evaluation = data.get('evaluation')  # 'successful', 'failed', or 'cancelled'
        notes = data.get('notes', '')
        user_notes = data.get('user_notes', '')
        
        if not trade_id or not evaluation:
            return jsonify({'status': 'error', 'message': 'معاملات مطلوبة مفقودة'})
        
        if evaluation not in ['successful', 'failed', 'cancelled']:
            return jsonify({'status': 'error', 'message': 'التقييم يجب أن يكون successful أو failed أو cancelled'})
        
        success = analyzer.evaluate_trade(trade_id, evaluation, notes, user_notes)
        
        if success:
            if evaluation == 'cancelled':
                return jsonify({
                    'status': 'success',
                    'message': 'تم إلغاء الصفقة'
                })
            else:
                return jsonify({
                    'status': 'success',
                    'message': f'تم تقييم الصفقة: {evaluation}'
                })
        else:
            return jsonify({'status': 'error', 'message': 'لم يتم العثور على الصفقة'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تقييم الصفقة: {str(e)}'})

@app.route('/api/trades/add', methods=['POST'])
def add_trade_for_evaluation():
    """إضافة صفقة للتقييم"""
    try:
        data = request.get_json(silent=True) or {}
        pair = data.get('pair')
        signal = data.get('signal')
        entry_price = data.get('entry_price')
        entry_time = data.get('entry_time')
        exit_time = data.get('exit_time')
        indicators_data = data.get('indicators_data', {})
        trade_type = data.get('trade_type', 'test')  # 'real' or 'test'
        
        if not all([pair, signal, entry_price, entry_time, exit_time]):
            return jsonify({'status': 'error', 'message': 'معاملات مطلوبة مفقودة'})
        
        trade_id = analyzer.add_trade_for_evaluation(
            pair, signal, entry_price, entry_time, exit_time, indicators_data, trade_type
        )
        
        return jsonify({
            'status': 'success',
            'trade_id': trade_id,
            'message': f'تم إضافة صفقة {trade_type} للتقييم'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في إضافة الصفقة: {str(e)}'})

@app.route('/api/ai/train-evaluations', methods=['POST'])
def train_ai_with_evaluations():
    """تدريب الذكاء الاصطناعي على تقييمات الصفقات"""
    try:
        data = request.get_json(silent=True) or {}
        trade_type = data.get('trade_type', 'real')
        success = analyzer.train_ai_with_evaluations(trade_type)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f"تم تدريب النماذج على تقييمات الصفقات بنجاح ({'حقيقية' if trade_type=='real' else 'تجريبية'})",
                'statistics': analyzer.get_trade_statistics(),
                'ai_metrics': analyzer.ai_system.get_performance_metrics()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f"فشل في تدريب النماذج لنوع '{'حقيقي' if trade_type=='real' else 'تجريبي'}' - تأكد من وجود 10+ تقييمات"
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في التدريب: {str(e)}'})

@app.route('/api/ai/training-status')
def get_training_status():
    """الحصول على حالة التدريب الحالية"""
    try:
        stats = analyzer.get_trade_statistics()
        ai_metrics = analyzer.ai_system.get_performance_metrics()
        
        # حساب نسبة التقدم
        total_trades = stats.get('total_trades', 0)
        pending_trades = stats.get('pending_evaluations', 0)
        evaluated_trades = total_trades - pending_trades
        
        progress_percentage = 0
        if total_trades > 0:
            progress_percentage = (evaluated_trades / total_trades) * 100

        # إحصائيات منفصلة لكل نوع
        real_trades = [t for t in analyzer.trade_history if t.get('trade_type') == 'real']
        test_trades = [t for t in analyzer.trade_history if t.get('trade_type') == 'test']
        real_eval = len([t for t in real_trades if t.get('user_evaluation') in ['successful','failed']])
        test_eval = len([t for t in test_trades if t.get('user_evaluation') in ['successful','failed']])
        real_pending = len([t for t in analyzer.pending_evaluations if t.get('trade_type') == 'real'])
        test_pending = len([t for t in analyzer.pending_evaluations if t.get('trade_type') == 'test'])

        training_status = {
            'overall': "جاهز للتدريب" if evaluated_trades >= 10 and pending_trades == 0 else (
                f"يحتاج {max(0, 10 - evaluated_trades)} تقييمات إضافية" if evaluated_trades < 10 else f"يوجد {pending_trades} صفقة معلقة"
            ),
            'real': {
                'evaluated': real_eval,
                'pending': real_pending,
                'can_train': real_eval >= 10 and real_pending == 0,
                'message': "جاهز للتدريب على الصفقات الحقيقية" if real_eval >= 10 and real_pending == 0 else (
                    f"يحتاج {max(0,10 - real_eval)} تقييمات حقيقية إضافية" if real_eval < 10 else f"يوجد {real_pending} صفقات حقيقية معلقة"
                )
            },
            'test': {
                'evaluated': test_eval,
                'pending': test_pending,
                'can_train': test_eval >= 10 and test_pending == 0,
                'message': "جاهز لتدريب المستخدم (تجريبي)" if test_eval >= 10 and test_pending == 0 else (
                    f"يحتاج {max(0,10 - test_eval)} تقييمات تجريبية إضافية" if test_eval < 10 else f"يوجد {test_pending} صفقات تجريبية معلقة"
                )
            }
        }
        
        return jsonify({
            'status': 'success',
            'training_status': training_status,
            'progress_percentage': round(progress_percentage, 1),
            'statistics': stats,
            'ai_metrics': ai_metrics,
            'recommendations': {
                'can_train_overall': evaluated_trades >= 10 and pending_trades == 0,
                'real_can_train': real_eval >= 10 and real_pending == 0,
                'test_can_train': test_eval >= 10 and test_pending == 0,
                'needs_more_data_overall': evaluated_trades < 10,
                'has_pending_overall': pending_trades > 0
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على حالة التدريب: {str(e)}'})

@app.route('/api/trades/analytics')
def get_trade_analytics():
    """الحصول على تحليلات مفصلة للصفقات"""
    try:
        stats = analyzer.get_trade_statistics()
        history = analyzer.get_trade_history(100)  # آخر 100 صفقة
        
        # تحليل الصفقات حسب الزوج
        pair_analysis = {}
        for trade in history:
            pair = trade.get('pair', 'Unknown')
            if pair not in pair_analysis:
                pair_analysis[pair] = {'total': 0, 'successful': 0, 'failed': 0}
            
            pair_analysis[pair]['total'] += 1
            if trade.get('user_evaluation') == 'successful':
                pair_analysis[pair]['successful'] += 1
            elif trade.get('user_evaluation') == 'failed':
                pair_analysis[pair]['failed'] += 1
        
        # حساب معدلات النجاح لكل زوج
        for pair in pair_analysis:
            total = pair_analysis[pair]['total']
            successful = pair_analysis[pair]['successful']
            if total > 0:
                pair_analysis[pair]['success_rate'] = round((successful / total) * 100, 2)
            else:
                pair_analysis[pair]['success_rate'] = 0
        
        # تحليل الصفقات حسب الإشارة
        signal_analysis = {'CALL': {'total': 0, 'successful': 0}, 'PUT': {'total': 0, 'successful': 0}}
        for trade in history:
            signal = trade.get('signal', 'Unknown')
            if signal in signal_analysis:
                signal_analysis[signal]['total'] += 1
                if trade.get('user_evaluation') == 'successful':
                    signal_analysis[signal]['successful'] += 1
        
        # حساب معدلات النجاح لكل إشارة
        for signal in signal_analysis:
            total = signal_analysis[signal]['total']
            successful = signal_analysis[signal]['successful']
            if total > 0:
                signal_analysis[signal]['success_rate'] = round((successful / total) * 100, 2)
            else:
                signal_analysis[signal]['success_rate'] = 0
        
        return jsonify({
            'status': 'success',
            'overall_stats': stats,
            'pair_analysis': pair_analysis,
            'signal_analysis': signal_analysis,
            'total_trades_analyzed': len(history),
            'analysis_date': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تحليل الصفقات: {str(e)}'})


@app.route('/api/results')
def get_results():
    """تشغيل التحليل عند الطلب"""
    try:
        if not analyzer.is_running or not hasattr(analyzer, 'analysis_config'):
            return jsonify({'status': 'error', 'message': 'التحليل غير مفعل حالياً'})

        config = analyzer.analysis_config
        results = {}
        current_time = datetime.now(timezone.utc) + timedelta(hours=2)

        for pair in config['pairs']:
            try:
                # تحويل الفترة إلى دقائق
                interval_str = f"{config['interval']}min"
                analysis = analyzer.analyze_pair(
                    pair,
                    config['period'],
                    config['indicators'],
                    interval_str,
                    config['interval']  # مدة الصفقة المختارة من المستخدم
                )
                if analysis:
                    # إصلاح التوقيت: الصفقة تبدأ بعد دقيقة من التحليل
                    next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
                    end_time = next_candle + timedelta(minutes=config['interval'])

                    analysis['trade_timing'] = {
                        'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'analysis_time': current_time.strftime('%H:%M:%S'),
                        'entry_time': next_candle.strftime('%H:%M:%S'),
                        'exit_time': end_time.strftime('%H:%M:%S'),
                        'duration': config['interval'],
                        'wait_time': '1 دقيقة'  # وقت الانتظار قبل الدخول
                    }

                    # إضافة الصفقة للتقييم لجميع الأزواج المختارة (للصفقات الحقيقية)
                    # استخدام النتيجة الأولى (API) كإشارة افتراضية
                    try:
                        if analysis.get('analyses') and len(analysis['analyses']) > 0:
                            first_analysis = analysis['analyses'][0]  # تحليل API
                            
                            # الحصول على السعر من أول تحليل أو بيانات الرسم البياني
                            entry_price = 0
                            if analysis.get('chart_data') and len(analysis['chart_data']) > 0:
                                entry_price = analysis['chart_data'][-1]['close']
                            
                            trade_id = analyzer.add_trade_for_evaluation(
                                pair=pair,
                                signal=first_analysis.get('signal', 'HOLD'),
                                entry_price=entry_price,
                                entry_time=first_analysis.get('entry_time', next_candle.strftime('%H:%M:%S')),
                                exit_time=first_analysis.get('exit_time', end_time.strftime('%H:%M:%S')),
                                indicators_data={},
                                trade_type='real'  # صفقة حقيقية
                            )
                            analysis['trade_id'] = trade_id
                            print(f"📝 تم إضافة صفقة حقيقية {pair} للتقييم: {trade_id}")
                        else:
                            print(f"⚠️ لا توجد نتائج تحليل لـ {pair}")
                    except Exception as e:
                        print(f"❌ خطأ في إضافة الصفقة للتقييم: {e}")

                    results[pair] = analysis
                else:
                    print(f"فشل تحليل {pair}")
            except Exception as e:
                print(f"خطأ في تحليل {pair}: {str(e)}")
                continue

        analyzer.latest_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis': results,
            'selected_indicators': config['indicators'],
            'status': 'success'
        }
        return jsonify(analyzer.latest_results)
        
    except Exception as e:
        print(f"خطأ في API results: {str(e)}")
        return jsonify({'status': 'error', 'message': f'خطأ في الخادم: {str(e)}'})
if __name__ == '__main__':
    print("🚀 بدء خادم التداول المتقدم...")
    print("📱 افتح المتصفح على: http://localhost:5000")
    print("📊 المؤشرات المتاحة:")
    print("   🔄 Trend: SMA, EMA, WMA, DEMA, TEMA, KAMA, HMA, T3")
    print("   ⚡ Momentum: RSI, STOCH, STOCHRSI, WILLR, MACD, PPO, ADX, CCI, MOM, ROC")
    print("   📈 Volatility: BBANDS, ATR, STDEV, DONCHIAN")
    print("   📊 Volume: OBV, CMF, AD, MFI, EMV, FI")
    print("   💰 Price: AVGPRICE, MEDPRICE, TYPPRICE, WCPRICE")
    print("   🎯 Misc: SAR, ULTOSC, TSI")
    app.run(debug=True, host='0.0.0.0', port=5000)