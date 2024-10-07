import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, Layer
from sklearn.model_selection import train_test_split
import numpy as np

# GPU 사용 확인
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 모든 GPU에 메모리 성장 옵션 설정 (필요한 만큼만 메모리 할당)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs are available: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Running on CPU.")

# 데이터 전처리 단계
def load_data():
    # 여기에 지식 그래프와 사용자-아이템 상호작용 데이터를 불러오는 코드를 작성하세요.
    # X_user, X_item은 사용자와 아이템의 ID, y는 상호작용 레이블입니다.
    # 임시 데이터 생성 (실제 데이터셋으로 대체)
    X_user = np.random.randint(0, 1000, size=(10000,))
    X_item = np.random.randint(0, 1000, size=(10000,))
    y = np.random.randint(0, 2, size=(10000,))  # 0 또는 1로 이루어진 레이블
    return X_user, X_item, y


X_user, X_item, y = load_data()

# 훈련과 테스트 데이터로 분리
X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
    X_user, X_item, y, test_size=0.2, random_state=42)


# KGAT 모델 클래스
class KGAT(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(KGAT, self).__init__()
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)

        # KGAT-specific layers: interaction layer
        self.attention = Dense(embedding_dim, activation='relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, item_input = inputs
        user_emb = self.user_embedding(user_input)
        item_emb = self.item_embedding(item_input)

        # Attention mechanism: user-item interaction
        interaction = self.attention(user_emb * item_emb)
        interaction = self.dropout(interaction)

        # Output prediction
        output = self.output_layer(interaction)
        return output


# 모델 파라미터
num_users = 1000  # 사용자 수
num_items = 1000  # 아이템 수
embedding_dim = 32  # 임베딩 차원

# 모델 생성
model = KGAT(num_users, num_items, embedding_dim)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 훈련 데이터로 모델 학습
model.fit([X_user_train, X_item_train], y_train, epochs=10, batch_size=64, validation_split=0.1)

# 테스트 데이터로 평가
test_loss, test_acc = model.evaluate([X_user_test, X_item_test], y_test)

print(f"테스트 정확도: {test_acc}")
