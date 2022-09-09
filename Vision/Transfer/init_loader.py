import pickle
import wandb
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models.resnet import BasicBlock, Bottleneck
import train
from train import Trainer
import random
import numpy as np
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data.distributed import DistributedSampler
torch.backends.cudnn.deterministic = True
import torch.distributed as dist
import torchvision.models as models
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)

def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

def init(conf=None , use_consistency_loss:bool = False, next_parameters:dict=None,model = None, block_to_friz=0):
	distributed = False

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wandb.login(key="ab95806c5c1d948bc6442948d8be591a6438c3fe")
	wandb.init(project='<Fashion_MNIST_Transfer>', entity='idori2', mode="online",
			   tags=[conf['dataset'], conf['model_conf']['model_name']])
	
	wandb.config.update(conf)
	loss_name = 'CrossEntropyLoss'

	wandb.config.update({"use_consistency_loss": use_consistency_loss})
	epochs_lr_decay = [conf['epochs'] // 3, conf['epochs'] * 2 // 3]

	model = model
	
	layers_from_end = next_parameters["num_layers_from_end"]
	layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'][layers_from_end:][::-1]
	wandb.config.update({"layers": layer_names})

	layers = []
	for layer_name in layer_names:
		layer = eval(f"model.{layer_name}")
		layers.append((layer_name, layer))

	if conf['input_ch'] == 1:
		model.conv1 = nn.Conv2d(conf['input_ch'], model.conv1.weight.shape[0], 3, 1, 1,
								bias=False)  # Small dataset filter size used by He et al. (2015)

		model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

	model = model.to(device)
	transform = transforms.Compose([transforms.Pad((conf['padded_im_size'] - conf['im_size']) // 2),
									transforms.ToTensor(),
									transforms.Normalize(tuple(conf['dataset_mean']), tuple(conf['dataset_std']))])

	
	train_dataset = datasets.FashionMNIST("Vision/data/FashionMNIST",train=True,download=True,transform = transform)

	#train_dataset = torch.utils.data.Subset(train_dataset,list(range(6000,54000)))
	g = torch.Generator()
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=conf['batch_size'], shuffle=True, generator=g)
	
	# ct = 0
	# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	# for child in model.children():
	# 	ct += 1
	# 	print(ct)
	# 	print(child)
	# 	print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	# 	if ct <= 8:
	# 		for param in child.parameters():
	# 			param.requires_grad = False
	# print(ct)

	if loss_name == 'CrossEntropyLoss':
		criterion = nn.CrossEntropyLoss()
		criterion_summed = nn.CrossEntropyLoss(reduction='sum')

	elif loss_name == 'MSELoss':
		criterion = nn.MSELoss()
		criterion_summed = nn.MSELoss(reduction='sum')

	momentum = 0.9
	if conf['dataset'] == 'ImageNet':
		weight_decay = 1e-4
	else:
		weight_decay = 5e-4
	optimizer = optim.SGD(model.parameters(),
						  lr=conf['model_conf']['lr'],
						  momentum=momentum,
						  weight_decay=weight_decay)


	trainer = Trainer(conf=conf, model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
			num_classes=conf['C'], device=device, epochs=conf['epochs'], use_consistency_loss=use_consistency_loss,
			layers=layers, distributed=distributed, next_parameters=next_parameters)

	return conf,model ,trainer, criterion_summed, device, conf['C'], conf['epochs'], epochs_lr_decay, conf["dataset"]
