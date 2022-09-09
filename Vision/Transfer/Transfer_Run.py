import sys
from analysis import Analyzer
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import datetime
import pickle
import wandb
import random
from losses import ConsistancyLoss
from init_loader import init
import argparse
import pickle as pick
import torch
import torchvision.models as models
import torchvision.datasets as datasets



def get_trained_model(conf):
	model = eval(f"models.{conf['model_conf']['model_name']}(pretrained=False, num_classes={conf['C']})")
	model.conv1 = torch.nn.Conv2d(conf['input_ch'],model.conv1.weight.shape[0],3,1,1,bias=False)
	model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
	model.load_state_dict(torch.load("Trained_Models/trained_model_MNIST_MAIN.pt"))
	return model


def data_loader_FMNIST(cinf):
	train_dataset = datasets.FashionMNIST("Vision/data/FashionMNIST",train=True,download=True)
	g = torch.Generator()
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=conf['batch_size'], shuffle=True, generator=g)
	

def main(conf, next_parameters):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	lr_decay = 0.1


	print(f"next_parameters:{next_parameters}")

	model = get_trained_model(conf)
	conf, model ,trainer, criterion_summed, device, num_classes, epochs, epochs_lr_decay, dataset = init(conf,
					use_consistency_loss=next_parameters['use_consistency_loss'], next_parameters=next_parameters,model=model)
	
	epoch_list = [1,20,40,50]
	layers = [4,7,11,13,15]
	layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
	eval_layers = []
	for layer_name in layer_names:
		layer = eval(f"model.{layer_name}")
		eval_layers.append((layer_name, layer))
	analyzer = Analyzer(conf, model, eval_layers, num_classes, device, criterion_summed)

	lr_scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer,
												  milestones=epochs_lr_decay,
												  gamma=lr_decay)
	
	cur_epochs = []
	layers = [5,6,7,8]
	for layer_fr in layers:
		model = get_trained_model(conf)
		conf, model ,trainer, criterion_summed, device, num_classes, epochs, epochs_lr_decay, dataset = init(conf,
					use_consistency_loss=next_parameters['use_consistency_loss'], next_parameters=next_parameters,model=model)
	
		epoch_list = [1,25,40,60,80,100]
		layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
		eval_layers = []
		for layer_name in layer_names:
			layer = eval(f"model.{layer_name}")
		eval_layers.append((layer_name, layer))
		analyzer = Analyzer(conf, model, eval_layers, num_classes, device, criterion_summed)

		lr_scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer,
												  milestones=epochs_lr_decay,
												  gamma=lr_decay)
	
		cur_epochs = []
		ct = 0
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		for child in model.children():
			ct += 1
			print(child)
			print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
			if ct <= layer_fr:
				for param in child.parameters():
					param.requires_grad = False
		print(ct)
		for epoch in range(1, epochs + 1):
			print(f"Starting epoch {epoch}")
			trainer.train(epoch)
			lr_scheduler.step()
			if epoch in epoch_list:
				cur_epochs.append(epoch)
				result = analyzer.analyze(epoch)
		torch.save(model.state_dict(),"Trained_Models/trained_model_on_FMNIST_100_epochs_first_"+str(layer)+"layers_frozen"+".pt")



def run_main(next_parameters):
	dataset_config = {'im_size': 28, 'padded_im_size': 32, 'C': 10, 'input_ch': 1, 'dataset': 'FashionMNIST', 'epochs': 100, 'batch_size': 256, 'model_conf': {'model_name': 'resnet18', 'lr': 0.00959692}, 'dataset_mean': [0.1307], 'dataset_std': [0.3081]}

	main(conf=dataset_config, next_parameters=next_parameters)
	

if __name__ == '__main__':
	alpha_const, layers_from_end, use_const = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'
	#config_path = sys.argv[4]
	next_parameters = {'alpha_consis': alpha_const,
					   'num_layers_from_end': layers_from_end, 'use_consistency_loss': use_const}
	run_main(next_parameters)
	