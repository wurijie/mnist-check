### 说明
这是一个使用PyTorch训练手写数字识别神经网络，并通过网页画板让你验证自己网络模型的准确性。你可以通过调整神经网络参数和训练过程提高模型识别精度，让枯燥的模型变得有趣起来！

### 依赖与环境
Python环境 Python3.6  
浏览器     Chrome  

- flask        1.1.1  
- torch        1.2.0  
- torchvision  0.4.0


### 使用方法
**安装依赖包**  
进入项目目录执行  
```
pip install -r requirement.txt
```  
  
**训练模型**
```
python my_mnist.py
```

**启动网站**
```
env FLASK_APP=check.py FLASK_DEBUG=1 flask run
```

**验证**  
访问127.0.0.1:5000，在黑色画板中按住鼠标左键写出数字，点击识别按钮识别数字。