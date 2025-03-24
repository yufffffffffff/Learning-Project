
#基础环境配置

#RGB颜色空间是通过对红、绿、蓝三种基色进行不同程度的加权混合来描述颜色的，而YCbCr颜色空间则是基于亮度和色度来描述颜色的。
#将RGB图像转换成YCbCr以便进一步对图像进行处理和分析

class args():

	# training args
	epochs = 30 #"number of training epochs, default is 2"
	batch_size = 8#8 #"batch size for training, default is 4"
	dataset_ir = "D:/KAIST/lwir"
	dataset_vi = "D:/KAIST/visible"     # 哥，好像全文并没有用到你，用的是replace()
	HEIGHT = 256
	WIDTH = 256

	# 作为训练网络的传入参数 这两个参数的值应该设置为多少呢？

	input_nc = 1
	output_nc = 1
	img_flag = 'L'#'RGB'#    #默认图片格式为黑白

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4 #"learning rate, default is 0.0001"
	lr_light = 1e-4  # "learning rate, default is 0.0001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	#resume = None#"./models/BTSFusion.model"      # 原先
	resume = "./models/BTSFusion.model"    # 更改


	# 你们三是什么用？
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	#model_path_gray = "./models/BTSFusion.model" #densefuse_gray.model"
	model_path_gray_test = "./models/test/test1_model.model"






