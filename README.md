# NJU_Auto
# 主车策略训练文件  
安装所需环境  
python==3.6  
ML-Agents==0.5.0  
tensorflow==1.7.1

Unity 版本 2018.3.10f1 (64-bit)  
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


模型放在hfreal\model\HFReal
19万个样本训练量
batch大小是512

一般在linux上训练，在win上看模型效果
num_timesteps是总训练步长

trainer.py中
总训练步长
num_timesteps = 2e9
如果训练后end，说明达到最大步长，需调高

alg\policies.py中
pdparam = tf.concat([pi, pi * 0.0 - 0.5], axis=1)
-0.5是logstd，表示对网络输出的动作采样的log方差，这个一开始训练设为-0.5就好，大概需要训练3e7步，3-4e7可以依次减小，可以根据buffer mean reward来判断，如果不怎么上升了，就可以减小，建议依次取-0.8, -1.2, -1.5, -1.8

unity build的文件放在本地src/HFReal/linux/HFreal下面
build完成后
HFReal 上传到Linux下HFReal/linux目录下

qrenv下面的 hfreal_path_xz.txt 上传到src的hfreal下面
QRSmartCar-v1.1\src\hfreal下面所有文件传到服务器中/home/yf/project/unity/hfreal


服务器训练：
trainer.py文件
	num_envs = 16
	nsteps=2**16
	train_model=True  不需要修改，if else实现
	env_path = "linux/" + dir + dir[:-1] + ".x86_64"

alg\policies.py文件
	a0 = self.pd.sample()


笔记本可视化推断：
num_envs = 1
nsteps=2**12
train_model=False   不需要修改，if else实现
env_path = None
alg\policies.py中
a0 = pi

开始训练：
第一次给文件夹权限
chmod -R 777 linux/

启动名字unity的python虚拟环境
source activate unity
运行QRSmartCar-v1.1\src\hfreal下面的train.py

训练结束杀死进程
ps -ef | grep python | grep -v grep | cut -c 9-15 | xargs kill -s 9

从checkpoint恢复训练
每个模型含有下面三个文件，将不需要的模型删除，checkpoint文件打开，第一行
.data-00000-of-00001
.index
.meta
