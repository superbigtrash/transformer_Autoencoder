import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# 0. Seed
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)
# =========================================================
# 1. Positional Encoding
# =========================================================
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()

        position = np.arange(sequence_length)[:, np.newaxis]
        dimension = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000,(2 * (dimension // 2)) / np.float32(d_model))
        angle_rads = position * angle_rates
        pos_encoding = np.zeros((sequence_length, d_model))

        # 짝수 index -> sin
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 홀수 index -> cos
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...],dtype=tf.float32)

    def call(self, x):
        return (x + self.pos_encoding[:, :tf.shape(x)[1], :])


# =========================================================
# 2. Transformer Encoder Block
# =========================================================
def transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1):
    # -----------------------------------------------------
    # Multi-Head Attention
    # -----------------------------------------------------
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query=x,value=x,key=x)
    attention_output = layers.Dropout(dropout)(attention_output)

    # -----------------------------------------------------
    # Add & Norm
    # -----------------------------------------------------
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # -----------------------------------------------------
    # Feed Forward Network
    # -----------------------------------------------------
    ffn_output = layers.Dense(ff_dim,activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)

    # -----------------------------------------------------
    # Add & Norm
    # -----------------------------------------------------
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    return x

# =========================================================
# 3. Transformer Autoencoder
# =========================================================
def build_transformer_autoencoder(
    sequence_length=50,
    n_features=3,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    latent_dim=16,
    num_encoder_blocks=2,
    dropout=0.1
):

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------
    inputs = layers.Input(shape=(sequence_length, n_features),name='input_sequence')

    # -----------------------------------------------------
    # Input Embedding
    # (sequence, 3)
    # ->
    # (sequence, d_model)
    # -----------------------------------------------------
    x = layers.Dense(d_model,nname='input_embedding')(inputs)

    # -----------------------------------------------------
    # Positional Encoding
    # -----------------------------------------------------
    x = PositionalEncoding(sequence_length=sequence_length,d_model=d_model)(x)

    # -----------------------------------------------------
    # Transformer Encoder
    # -----------------------------------------------------
    for _ in range(num_encoder_blocks):
        x = transformer_encoder(x=x,d_model=d_model,num_heads=num_heads,ff_dim=ff_dim,dropout=dropout)
        
    # -----------------------------------------------------
    # Bottleneck
    # -----------------------------------------------------
    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(32,activation='relu',name='encoder_dense')(x)
    latent = layers.Dense(latent_dim,activation=None,name='latent_vector')(x)

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------
    x = layers.Dense(sequence_length * d_model,activation='relu',name='decoder_dense')(latent)
    x = layers.Reshape((sequence_length, d_model),name='decoder_reshape')(x)
    x = layers.Dense(32,activation='relu',name='decoder_ffn')(x)

    # -----------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------
    outputs = layers.Dense(n_features,activation=None,name='reconstruction')(x)
    model = Model(inputs=inputs,outputs=outputs,name='Transformer_Autoencoder')

    return model


# 4. Sequence 생성 함수
def make_sequence(data,labels,sequence_length):

    X = []
    Y = []

    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])

        # window 마지막 시점의 label
        Y.append(labels[i + sequence_length - 1])

    return (np.array(X),np.array(Y))


# 5. Dataset Load
normal = pd.read_csv('./press_data_normal.csv',index_col=0)
outlier = pd.read_csv('./outlier_data.csv',index_col=0)

normal_data = normal.copy()
outlier_data = outlier.copy()


use_col = ['AI0_Vibration','AI1_Vibration','AI2_Current']


# 6. Feature / Label
X_normal = normal_data[use_col]
y_normal = normal_data['Equipment_state']

X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['Equipment_state']

print("Normal shape :",X_normal.shape)
print("Anomaly shape :",X_anomaly.shape)

# =========================================================
# 7. 정상 데이터 Train / Test 분리
#
# 정상 앞부분 15000개:
# Autoencoder 학습
#
# 정상 뒷부분:
# Validation / Test

X_train_normal = X_normal.iloc[:15000]
y_train_normal = y_normal.iloc[:15000]

X_normal_rest = X_normal.iloc[15000:]
y_normal_rest = y_normal.iloc[15000:]

# 이상 데이터
X_anomaly_all = X_anomaly
y_anomaly_all = y_anomaly


# =========================================================
# 8. Scaling
#
# 매우 중요:
# scaler는 정상 Train 데이터에만 fit
# =========================================================
scaler = MinMaxScaler()


X_train_scaled = scaler.fit_transform(X_train_normal)
X_normal_rest_scaled = scaler.transform(X_normal_rest)
X_anomaly_scaled = scaler.transform(X_anomaly_all)

# =========================================================
# 9. Sequence 생성
# =========================================================
SEQUENCE_LENGTH = 50

X_train, Y_train = make_sequence(X_train_scaled,np.array(y_train_normal),SEQUENCE_LENGTH)
X_normal_seq, Y_normal_seq = make_sequence(X_normal_rest_scaled,np.array(y_normal_rest),SEQUENCE_LENGTH)
X_anomaly_seq, Y_anomaly_seq = make_sequence(X_anomaly_scaled,np.array(y_anomaly_all),SEQUENCE_LENGTH)

print()
print("X_train shape :",X_train.shape)
print("Normal sequence shape :",X_normal_seq.shape)
print("Anomaly sequence shape :",X_anomaly_seq.shape)

# =========================================================
# 10. Validation / Test 분리
# =========================================================
# ---------------------------------------------------------
# 정상 데이터
# 20% -> Validation
# 80% -> Test
# ---------------------------------------------------------
normal_valid_size = int(len(X_normal_seq) * 0.2)
X_valid_normal = (X_normal_seq[:normal_valid_size])
Y_valid_normal = (Y_normal_seq[:normal_valid_size])

X_test_normal = (X_normal_seq[normal_valid_size:])
Y_test_normal = (Y_normal_seq[normal_valid_size:])

# ---------------------------------------------------------
# 이상 데이터
#
# 앞 50%:
# threshold 확인 또는 validation 분석용
#
# 뒤 50%:
# 최종 test
# ---------------------------------------------------------
anomaly_valid_size = int(len(X_anomaly_seq) * 0.5)
X_valid_anomal = (X_anomaly_seq[:anomaly_valid_size])
Y_valid_anomal = (Y_anomaly_seq[:anomaly_valid_size])

X_test_anomal = (X_anomaly_seq[anomaly_valid_size:])
Y_test_anomal = (Y_anomaly_seq[anomaly_valid_size:])

# =========================================================
# 11. 최종 Test Dataset
# =========================================================
X_test = np.vstack((X_test_normal,X_test_anomal))
Y_test = np.hstack((Y_test_normal,Y_test_anomal))

print()
print("Train          :",X_train.shape)
print("Valid Normal   :",X_valid_normal.shape)
print("Valid Anomaly  :",X_valid_anomal.shape)
print("Test Normal    :",X_test_normal.shape)
print("Test Anomaly   :",X_test_anomal.shape)
print("Final Test     :",X_test.shape)

# =========================================================
# 12. Transformer Autoencoder 설정
# =========================================================
N_FEATURES = 3
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
LATENT_DIM = 16
NUM_ENCODER_BLOCKS = 2
DROPOUT = 0.1

# =========================================================
# 13. Model 생성
# =========================================================
model = build_transformer_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    latent_dim=LATENT_DIM,
    num_encoder_blocks=NUM_ENCODER_BLOCKS,
    dropout=DROPOUT
)
# =========================================================
# 14. Compile
# =========================================================
model.compile(optimizer=Adam(learning_rate=1e-3),loss='mse')
model.summary()

# =========================================================
# 15. Callback
# =========================================================
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6,verbose=1)

# =========================================================
# 16. Train
#
# Autoencoder:
#
# X_train -> X_train
#
# 정상 데이터만 학습
# =========================================================
history = model.fit(X_train,X_train,validation_data=(X_valid_normal,X_valid_normal),
    epochs=100,batch_size=64,shuffle=False,
    callbacks=[early_stopping,reduce_lr],verbose=1)

# =========================================================
# 17. Training Loss Plot
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Transformer Autoencoder Training Loss')
plt.legend()
plt.grid()
plt.show()
# =========================================================
# 18. 정상 Validation Reconstruction
# =========================================================
X_valid_pred = model.predict(X_valid_normal,verbose=0)
# =========================================================
# 19. 정상 Validation Reconstruction Error
# 각 Window마다 MSE 1개 계산
# =========================================================
valid_error = np.mean(np.square(X_valid_normal- X_valid_pred),axis=(1, 2))

print()
print("Validation reconstruction error")
print("Mean :",np.mean(valid_error))
print("Std  :",np.std(valid_error))
print("Min  :",np.min(valid_error))
print("Max  :",np.max(valid_error))

# =========================================================
# 20. Threshold 설정
#
# 정상 Validation reconstruction error의
# 상위 1%를 이상으로 간주
# =========================================================
threshold = np.percentile(valid_error,99)
print()
print("Threshold :",threshold)

# =========================================================
# 21. Normal Test Reconstruction
# =========================================================
normal_pred = model.predict(X_test_normal,verbose=0)
normal_error = np.mean(np.square(X_test_normal- normal_pred),axis=(1, 2))

# =========================================================
# 22. Anomaly Test Reconstruction
# =========================================================
anomaly_pred = model.predict(X_test_anomal,verbose=0)
anomaly_error = np.mean(np.square(X_test_anomal- anomaly_pred),axis=(1, 2))

# =========================================================
# 23. Reconstruction Error 비교
# =========================================================
print()
print("Normal Test Error")
print("Mean :",np.mean(normal_error))
print("Max  :",np.max(normal_error))

print()
print("Anomaly Test Error")
print("Mean :",np.mean(anomaly_error))
print("Min  :",np.min(anomaly_error))

# =========================================================
# 24. Reconstruction Error Distribution
# =========================================================
plt.figure(figsize=(12, 5))
plt.hist(normal_error,bins=100,alpha=0.6,label='Normal')
plt.hist(anomaly_error,bins=100,alpha=0.6,label='Anomaly')
plt.axvline(threshold,linestyle='--',label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 25. Test 전체 Reconstruction Error
# =========================================================
X_test_pred = model.predict(X_test,verbose=0)
test_error = np.mean(np.square(X_test- X_test_pred),axis=(1, 2))
# =========================================================
# 26. Threshold 기준 Prediction
#
# 정상 = 0
# 이상 = 1
# =========================================================
Y_pred = (test_error > threshold).astype(int)

# =========================================================
# 27. Label 확인
#
# Equipment_state가
# 정상=0 / 이상=1 이라는 전제
# =========================================================
Y_test = Y_test.astype(int)

# =========================================================
# 28. 평가
# =========================================================
print()
print("==============================")
print("Evaluation")
print("==============================")

print()
print("Accuracy :",accuracy_score(Y_test,Y_pred))
print("Precision :", precision_score(Y_test,Y_pred,zero_division=0))
print("Recall :", recall_score(Y_test,Y_pred,zero_division=0))
print("F1 Score :",f1_score(Y_test,Y_pred,zero_division=0))

# Reconstruction error 자체를
# anomaly score로 사용
try:
    auc = roc_auc_score(Y_test,test_error)
    print("ROC-AUC :",auc)
    
except ValueError : 
    print("ROC-AUC 계산 불가")

# =========================================================
# 29. Classification Report
# =========================================================
print()
print(classification_report(Y_test,Y_pred,digits=4,zero_division=0))

# =========================================================
# 30. Confusion Matrix
# =========================================================
cm = confusion_matrix(Y_test,Y_pred)
print()
print("Confusion Matrix")
print(cm)

# =========================================================
# 31. Confusion Matrix Plot
# =========================================================
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i, j],ha='center',va='center')

plt.xticks([0, 1],['Normal','Anomaly'])
plt.yticks([0, 1],['Normal','Anomaly'])
plt.show()

# =========================================================
# 32. 시간 순 Reconstruction Error Plot
# =========================================================
plt.figure(figsize=(14, 5))
plt.plot(test_error,label='Reconstruction Error')
plt.axhline(threshold,linestyle='--',label='Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction Error')
plt.title('Anomaly Score over Time')
plt.legend()
plt.grid()
plt.show()