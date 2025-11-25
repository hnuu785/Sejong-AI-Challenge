import random
import torch
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
train_x = pd.read_csv('../data/train_x.csv')
train_y = pd.read_csv('../data/train_y.csv')
test_x = pd.read_csv('../data/test_x.csv')

print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)
print("Test X shape:", test_x.shape)

# 특성과 타겟 분리
X_train = train_x[['fish_length', 'fish_weight']].values
y_train = train_y['target'].values
X_test = test_x[['fish_length', 'fish_weight']].values

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# Submission 파일 생성
submission = pd.DataFrame({
    'id': test_x.index,
    'target': y_pred
})

submission.to_csv('../data/submission.csv', index=False)
print("\nSubmission file created!")
print(f"Predicted {len(y_pred)} samples")
print(f"Class distribution: {np.bincount(y_pred)}")
print("\nFirst few predictions:")
print(submission.head())