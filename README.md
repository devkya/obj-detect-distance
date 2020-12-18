# 시각장애인을 위한 장애물 탐지 서비스
## 개발환경
* 기본적인 개발 환경 설정은 tensorflow object detection examples 과 동일
* [여기서 확인](https://github.com/Devkya/obj-detect-distance/tree/master/lite/examples/object_detection/android)

## 설명
* 시각장애인을 위한 장애물을 탐지하고 장애물 거리 측정 서비스를 구현.
* 거리 측정 알고리즘은 psmNet을 사용하고 싶었으나 시간적 한계로 실패.  
* [distance measurement algorithm](http://emaraic.com/blog/distance-measurement)  
* 기본 틀은 tensorflow homepage의 object-detection examples 사용.  
* 음성과 거리 측정 알고리즘만 탑재함.  

# TensorFlow Examples

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png" /><br /><br />
</div>

<h2>Most important links!</h2>

* [Community examples](./community)
* [Course materials](./courses/udacity_deep_learning) for the [Deep Learning](https://www.udacity.com/course/deep-learning--ud730) class on Udacity

If you are looking to learn TensorFlow, don't miss the
[core TensorFlow documentation](http://github.com/tensorflow/docs)
which is largely runnable code.
Those notebooks can be opened in Colab from
[tensorflow.org](https://tensorflow.org).

<h2>What is this repo?</h2>

This is the TensorFlow example repo.  It has several classes of material:

* Showcase examples and documentation for our fantastic [TensorFlow Community](https://tensorflow.org/community)
* Provide examples mentioned on TensorFlow.org
* Publish material supporting official TensorFlow courses
* Publish supporting material for the [TensorFlow Blog](https://blog.tensorflow.org) and [TensorFlow YouTube Channel](https://youtube.com/tensorflow)

We welcome community contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) and, for style help,
[Writing TensorFlow documentation](https://www.tensorflow.org/community/documentation)
guide.

To file an issue, use the tracker in the
[tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/new?template=20-documentation-issue.md) repo.

## License

[Apache License 2.0](LICENSE)
