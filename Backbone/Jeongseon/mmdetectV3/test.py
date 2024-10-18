import torch

# 가중치 파일 로드
checkpoint = torch.load('efficientdet-d8.pth', map_location='cpu')

# 최상위 키 확인
print("Top-level keys:", checkpoint.keys())

# 모델 가중치 키 확인
if 'model' in checkpoint:
    model_keys = checkpoint['model'].keys()
    print("\nNumber of keys in model:", len(model_keys))
    print("\nSample keys:")
    for key in list(model_keys)[:10]:  # 처음 10개 키만 출력
        print(key)

# 가중치 형태 확인
print("\nShape of some tensors:")
for key in list(model_keys)[:5]:
    print(f"{key}: {checkpoint['model'][key].shape}")