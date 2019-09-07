#--*-- encoding: UTF-8 --*--
import torch
from PIL import Image
import numpy as np
from flask import Flask, request, render_template
import base64
from my_mnist import CNN


#-------下面部分为web容器代码-------
app = Flask(__name__)

#用于接收手写的图片
@app.route('/cnn', methods=["POST"])
def test():
    imgdata =  base64.b64decode(request.form['img'][22:])
    with open('temp.png', 'wb') as fd:
        fd.write(imgdata)

    # 通过之前的模型识别用户手写的图片
    return transfer()

#返回index.html，用户书写数字页面
@app.route('/')
def index():
    return render_template(
        'index.html'
    )


#-------下面部分为神经网络验证部分----------
#使用my_mnist中定义的神经网络
cnn = CNN()
cnn.load_state_dict(torch.load('net_params.pkl')) #加载训练的模型结果

def transfer():
    #将网页手写的图片转换成28*28，并转成单通道灰度图片
    np_data = np.array(Image.open('./temp.png').resize((28, 28),Image.ANTIALIAS).convert('L'))
    t_data = torch.from_numpy(np_data).unsqueeze(dim=0).unsqueeze(dim=0).type(torch.FloatTensor)/255.0

    predict = cnn(t_data)

    num = torch.max(predict, 1)[1].numpy()
    return str(num[0])