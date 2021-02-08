# NJU_Auto
# 主车策略训练文件  
安装所需环境  
python==3.6  
ML-Agents==0.5.0  
tensorflow==1.7.1

Unity 版本 2018.3.10f1 (64-bit)  
https://unity.cn/releases  
unity 项目文件下载地址
链接：https://pan.baidu.com/s/19r_djIuLh2e6hma5f1Zfrw 
提取码：hp3s 


解压文件 

unity项目文件位置  
SmartCar\Env  
使用unity打开以上路径项目  
在unity中选择  
Env\Assets\HFReal  
![image](https://github.com/buaazeus/NJU_Auto/blob/main/images/1.png)  

unity中需安装ML-Agents  
项目文件中已经包含了ML-Agents-0.5.0，可以跳过以下步骤  
ML-Agents安装方法   
可在菜单栏，window--package manager中安装，也可以单独下载离线安装，安装步骤见为  
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md  


训练代码路径  
SmartCar\src\hfreal  
trainer.py为主训练文件  
在SmartCar\src\hfreal路径下运行即可开始训练/推断  
python trainer.py
等待终端中提示press start button， 此时需在unity中按开始按钮，正常的话车辆就开始训练了。  
在linux下训练时，将SmartCar\src\hfreal文件夹上传即可    

模型文件放在SmartCar\src\hfreal\model\HFReal
目前文件默认模型为5456384  
可以在上面的路径中选择其他模型  
通过修改checkpoint文件第一行model_checkpoint_path: "-5456384"即可
如果需要从头开始重新训练，把checkpoint文件删除即可  
每个模型含有下面三个文件  
.data-00000-of-00001  
.index  
.meta  

# 生成linux训练环境文件
由于训练环境是在windows下unity中创建的  
训练时需要在linux服务器上进行  
所以需要生成linux训练环境文件  
由于第一次训练需要生成路径文件，需在windows下进行，生成的文件名为hfreal_path_xz.txt，存放在SmartCar\Env\下，将hfreal_path_xz.txt传至linux中的hfreal\下面即可  
打开hfreal场景，菜单栏File--build settings，按下图选择，然后build，选择路径，例如SmartCar\src\hfreal\linux\HFReal：  
![image](https://github.com/buaazeus/NJU_Auto/blob/main/images/2.png)  
build完成后，会生成HFReal_Data文件夹和HFReal.x86_64文件，这两部分需打包上传至linux服务器，例如可以上传至hfreal/linux  
第一次训练前需给文件夹运行权限  
chmod -R 777 linux/  
然后就可以在linux服务器上进行训练了，训练生成的模型会保存在路径  
将模型下载至windows下，可以进行推断，就可以观察模型的实际效果  
（windows下也可以进行训练，受限于机器性能，训练速度较慢）  

# trainer.py文件内容介绍  
![image](https://github.com/buaazeus/NJU_Auto/blob/main/images/3.png)  
num_envs为同时开启环境数量，在linux服务器训练时，可设定为16，在windows推断时，设定为1.  
env_path为环境路径，在linux服务器训练时采用第一行，在windows推断时，采用第二行None  

# alg\policies.py介绍  
![image](https://github.com/buaazeus/NJU_Auto/blob/main/images/4.png)  
pdparam = tf.concat([pi, pi * 0.0 - 0.5], axis=1)  
-0.5是log std，表示对网络输出的动作采样的log 方差，这个一开始训练设为-0.5就好，大概需要训练3e7步，3-4e7可以依次减小，可以根据buffer mean reward来判断，如果不怎么上升了，就可以减小，建议依次取-0.8, -1.2, -1.5, -1.8  
a0 = self.pd.sample()，使用带有探索的策略，在linux上训练时使用   
a0 = pi，确定的最优策略，在windows推断时使用  



