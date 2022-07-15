# Requirements
- pip install -r requirements.txt

# 테스트 데이터 셋 성능 테스트 
### 패션 스타일 분류 
- python run/test_style.py

### 패션 특성 분류 
- python run/test_attribute.py

# 테스트 성능 확인
## 전체 테스트 데이터셋의 성능 수치 확인 
### 패션 스타일 분류 정확도
```
C:\Users\user\.conda\envs\gpu_cuda\python.exe C:/Users/user/Sunghoon_Workspace/Aim/kfashoin_ai_model/run/test.py
Starting..
=> loading checkpoint '../checkpoint/kfashion_style/model_best.pth.tar'
Test: 100%|??????????| 322/322 [02:07<00:00,  2.52it/s]
Top-3 recall: 0.9111
Top-5 recall: 0.9889
[DONE] Total time spent :131.4260
```


### 패션 특성 분류 정확도
```
C:\Users\user\.conda\envs\gpu_cuda\python.exe C:/Users/user/Sunghoon_Workspace/Aim/kfashoin_ai_model/run/test2.py
Starting..
=> loading checkpoint '../checkpoint/kfashion_category/model_category_best.pth.tar'
Test: 100%|??????????| 428/428 [03:29<00:00,  2.04it/s]
Top-3 recall: 0.934
[DONE] category time spent :212.9350
=> loading checkpoint '../checkpoint/kfashion_detail/model_detail_best.pth.tar'
Test: 100%|??????????| 255/255 [02:21<00:00,  1.80it/s]
Top-3 recall: 0.6379
[DONE] detail time spent :355.6120
=> loading checkpoint '../checkpoint/kfashion_texture/model_texture_best.pth.tar'
Test: 100%|??????????| 410/410 [03:12<00:00,  2.13it/s]
Top-3 recall: 0.8333
[DONE] texture time spent :548.7300
=> loading checkpoint '../checkpoint/kfashion_print/model_print_best.pth.tar'
Test: 100%|??????????| 353/353 [02:52<00:00,  2.05it/s]
Top-3 recall: 0.9478
[DONE] print time spent :721.7176
Average recall: 0.83825
[DONE] Total time spent :721.7176
```

## 개별 데이터셋에 대한 성능 수치 확인
### 패션 스타일 분류
- 아래 파일들을 통해 각 row 별 예측 결과를 확인 
```
output/style_correct_3.csv
output/style_correct_5.csv
output/style_wrong_3.csv
output/style_wrong_5.csv
```

### 패션 특성 분류
- 아래 파일들을 통해 각 row 별 예측 결과를 확인
```
output/category_correct.csv
output/category_wrong.csv
output/detail_correct.csv
output/detail_wrong.csv
output/print_correct.csv
output/print_wrong.csv
output/texture_correct.csv
output/texture_wrong.csv
```

## 실제 이미지에 대해 모델이 출력한 결과와 비교
### 패션 스타일 분류 모델이 잘못 예측한 실제 이미지 샘플
- output/style_correct_5.csv --> output/wrong_sample/PRS_796_41.jpg
- output/style_wrong_5.csv --> output/wrong_sample/TimelessW_086_56.jpg

### 패션 특성 분류 모델이 잘못 예측한 실제 이미지 샘플
- output/category_wrong.csv--> output/wrong_sample/143394_아우터.jpg
- output/detail_wrong.csv --> output/wrong_sample/48076_상의.jpg
- output/print_wrong.csv --> output/wrong_sample/66485_하의.jpg
- output/texture_wrong.csv --> output/wrong_sample/77845_원피스.jpg

# 시험수행 과정 및 결과 로그 txt파일
- output/log/테스트로그.txt
- output/log/테스트환경.txt
- output/log/테스트환경.PNG


