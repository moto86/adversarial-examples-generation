# adversarial-examples-generation
[foolbox](https://github.com/bethgelab/foolbox)で敵対的画像を生成するサンプルコードです．
FGSM，C&Wアタック，Deepfoolのサンプルです．
kerasベースで，使い方は以下です．

```python
adversarial_generator.py -m model_file -a attack_type
```
