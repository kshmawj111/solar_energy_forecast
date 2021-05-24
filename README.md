# solar_energy_forcast
 
Dacon Link: https://dacon.io/competitions/official/235680/overview/description/

# 태양열 발전량 시계열 예측 대회

### 개발환경:
   
   1) Windows 10 Home - AMD ryzen 2700x, GTX 1060 6GB
   2) Google Colab - for notebooks
   3) "Deepo" Docker container

### 사용한 패키지

 1) MxNet gluonts for modelling
 2) Scikit Learn for baseline code


### 참고 논문

1) MQ-RNN : https://arxiv.org/pdf/1711.11053.pdf
2) DeepAR : https://arxiv.org/abs/1704.04110
3) DeepVAR : https://arxiv.org/abs/2006.08338


### 참고 사이트

1) GluonTS : https://ts.gluon.ai/
2) Amazon Sagemaker : https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html


### 수행 과정

1) Multivariate Time Series Forecasting에 관한 대회였다. 3개년의 학습 데이터가 주어지고 각 테스트 셋마다 7일치의 데이터가 주어지며 해당 테스트 데이터로 이후 이틀치의 발전량을 예측하는 것이 목표이다. EDA를 통해 DHI, DNI가 매우 중요함을 확인하였고 DHI가 발전량의 최저 수준을 결정하는 것을 확인하였다. 최대 발전량은 DHI와 DNI의 어떠한 조합에 의하여 결정되었다.

2) Pinball-loss라는 Quantile 값에 대한 예측을 해야하는 대회였고 이를 위해 PDF를 형성해 줄 수 있는 모델이 좋다고 판단했다. PDF만 알면 샘플링으로 quantile 지점에 대한 값을 뽑아 내면 되기 때문이다.

3) Baseline은 LGBM을 사용하여 모델링 되었지만 뭔가 LGBM을 사용하는 것이 잘 납득이 가지 않아 다른 모델을 찾아보았다.

4) 제일 먼저 발견한 것은 MQ-RNN이라는 모델이었고 해당 모델을 조금 수정하여 Input Layer에 FCC Layer나 Convolution Layer를 덧대어 Feature 벡터를 입력으로 받도록 할 생각이었다. 나아가 MQ-RNN이 Seq2Seq 구조로 되어있음을 확인하여 중간에 Attention Layer를 추가하여 Decoder 레이어에서 조금 더 나은 결과를 내도록 하려고 하였다. 한 가지 문제는 연산량이 과도하게 많아져 시간이 좀 오래 걸릴 것 같았다.

5) 수정할 방법을 생각하는 중에 아마존의 DeepAR이라는 Deep Learning 기반 Auto Regressive한 모델을 찾아 해당 모델을 사용하기로 하였다.

6) GluonTS 공식 문서와 아마존 Sage Maker에 탑재된 문서를 보며 패러미터 등을 이해하고 내부 코드를 들여다보며 대강 어떤 식으로 동작하는지 이해했다.

7) DeepAR의 경우 Multivariate Feature를 사용하기 위해서 사용하려는 시점의 Feature 값들이 필요한 것을 확인하여 맨 테스트 셋에 피쳐를 넣기 위한 작업을 수행하였다. 해당 과정은 src/feature_maker.py에 기록되어 있다. 해당 모듈은 대회가 종료 된 후에 구조가 너무 엉망진창으로 되어 있어서 따로 refactoring을 하였다. 대회에서 좋은 결과를 얻지 못하였지만 refactoring을 통해 Abstraction과 Class의 역할 구분과 프로그램, 모듈의 설계가 얼마나 중요한지 다시금 깨닫았다.

8) 모델을 튜닝은 Bayesian Optimization을 통하여 진행하였고 이후 Submission을 제출했으나 좋지 않은 결과를 얻었다.

9) 이후에 Trainig-set으로부터 임의로 Validation-set을 대회의 Test-set 처럼 만든 뒤에 해당 Validation-set에 대한 평가와 plot을 찍어보며 어떤 방법으로 Test-set에 피쳐를 채워 넣는 것이 중요한지 실험해보았다. 그 결과 과거 데이터에서 학습할 구간의 Feature 데이터와 가장 유사한 구간의 Feature 구간 중에서 이틀 치의 피쳐 값을 가져오는게 제일 좋았었다.

10) 그러나 결과의 진전은 크게 나지 않았고 대회가 끝날 때 즈음, 팀원 중  진정한 의미의 Multivariate Feature를 지원하는 DeepVAR의 존재를 알게 되어 황급히 해당 모델로 변경하였다.

11) 그러나 시간이 너무 지체되어 해당 모델로 제출하지 못하였고 아쉬움을 남긴채 종료가 되었다.

12) 아쉬운 마음이 많이 남아 4/7일부터 진행되는 새로운 태양열 발전량 예측 대회에 DeepVAR 모델을 통해 진행하려고 한다.
