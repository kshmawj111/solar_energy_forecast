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
3) DeepVAR : https://arxiv.org/abs/1910.03002


### 참고 사이트

1) GluonTS : https://ts.gluon.ai/
2) Amazon Sagemaker : https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html


### 수행 과정

1) Multiple Time Series Forecasting에 관한 대회였다. 3개년의 학습 데이터가 주어지고 각 테스트 셋마다 7일치의 데이터가 주어지며 해당 테스트 데이터로 이후 이틀치의 발전량을 예측하는 것이 목표이다. EDA를 통해 DHI, DNI가 매우 중요함을 확인하였고 DHI가 발전량의 최저 수준을 결정하는 것을 확인하였다. 최대 발전량은 DHI와 DNI의 어떠한 조합에 의하여 결정되었다. 이 과정은 jupyter-notebook으로 작성되었다.

2) Pinball-loss라는 Quantile 값에 대한 예측을 해야하는 대회였고 이를 위해 PDF를 형성해 줄 수 있는 모델이 좋다고 판단했다. PDF만 알면 샘플링으로 quantile 지점에 대한 값을 뽑아 내면 되기 때문이다.

3) Baseline은 LGBM을 사용하여 모델링 되었지만 뭔가 LGBM을 사용하는 것이 잘 납득이 가지 않아 다른 모델을 찾아보았다.

4) 제일 먼저 발견한 것은 MQ-RNN이라는 모델이었고 해당 모델을 조금 수정하여 Input Layer에 FCC Layer나 Convolution Layer를 덧대어 Feature 벡터를 입력으로 받도록 할 생각이었다. 나아가 MQ-RNN이 Seq2Seq 구조로 되어있음을 확인하여 중간에 Attention Layer를 추가하여 Decoder 레이어에서 조금 더 나은 결과를 내도록 하려고 하였다. 한 가지 문제는 연산량이 과도하게 많아져 시간이 좀 오래 걸릴 것 같았다.

5) 수정할 방법을 생각하는 중에 아마존의 DeepAR이라는 Deep Learning 기반 Auto Regressive한 모델을 찾아 해당 모델을 사용하기로 하였다. 이 당시에는 수학적 지식이 거의 없어 논문을 이해하는데 상당히 힘들었다. 수학적 지식을 쌓아야 함을 절실히 깨닫고 나서 확률과 통계, 선형대수에 대한 강의를 찾아 듣고서 근래에 다시 논문을 읽으니 이전보다 훨씬 더 잘 읽혔고 이해도 더 많이 갔다. 수학은 지속적으로 공부해야할 동반자임을 깨닫았다.

6) 모델은 GluonTS 공식 문서와 아마존 Sage Maker에 탑재된 문서를 보며 패러미터 등을 이해하고 내부 코드를 들여다보며 어떤 식으로 동작하는지 이해했다. 다만, 그 내용이 상당히 불친절하고 별로 없어서 내부 코드와 클래스를 직접 디버깅하고 찾아가며 이해하였다. 모델을 최소한의 수준으로 이해하는데까지 약 이틀은 걸렸던 것 같다. MxNet을 아예 처음보기 때문에 그런 것도 있다.

7) DeepAR의 경우 Multiple Feature를 사용하기 위해서 사용하려는 시점의 Feature 값들이 필요한 것을 확인하여 맨 테스트 셋에 피쳐를 넣기 위한 작업을 수행하였다 실험적으로 Mean, Median, MA와 테스트 시점과 가장 차이가 적은 시점의 이틀 뒤의 데이터를 복사하는 방법을 사용하였다.

8) 각 방법에 대한 평가를 해야 했는데 기능들이 너무 따로 따로 작성 되어서 테스트 하기도 쉽지 않았다. 전부 뜯어 고쳐서 One line execution을 목표로 하여 각 클래스를 생성하고 통합하는 작업을 진행했다. 또한, 시계열 데이터이기 때문에 그냥 단순히 train_test_split은 당연히 적용할 수 없었고 특정한 날짜를 샘플링하여 해당 날짜로부터 9일치에 대한 데이터를 원본에서 추출해야 했다. 이 일련의 과정들이 너무 너저분하고 복잡하게 얽혀 있어서 교통정리를 해야겠다는 생각으로 통합하는 작업을 시행했다.

9) 이후 실험 중에 GluonTS 중 DeepVAR이란 패키지가 있음을 확인받고 해당 모델로 변경했다. 기존 feature를 prediction range에 임의로 추가하는 작업은 시간이 오래걸릴 뿐만 아니라 제대로 학습이 되는 결과인지도 불확실했기 때문이다. 그 결과 DeepAR보다 피처의 변화가 급작스러운 부분, 예를 들면, 비가 와서 일조량이 줄어드는 경우,를 조금 더 잘 찾아내서 예측할 수 있었다.

### DeepAR의 결과 예시 (주피터 노트북에도 예시 결과를 첨부하였다.)

![image](https://user-images.githubusercontent.com/45940359/119340097-00a40980-bccd-11eb-82f3-dd3f5ff831b0.png) 

