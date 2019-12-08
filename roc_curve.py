from __future__ import print_function, division

import torch
import torch.nn as nn
import argparse
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image as pil_image
from tqdm import tqdm
from network.classifier import *
from network.transform import mesonet_data_transforms

def preprocess_image(image, cuda=True):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	preprocess = mesonet_data_transforms['test']
	preprocessed_image = preprocess(pil_image.fromarray(image))

	preprocessed_image = preprocessed_image.unsqueeze(0)
	if cuda:
		preprocessed_image = preprocessed_image.cuda()
	return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
	preprocessed_image = preprocess_image(image, cuda)
	output = model(preprocessed_image)
	output = post_function(output)
	#print(float(output[0][0].cpu().detach().numpy()))
	prediction = float(output[0][1].cpu().detach().numpy())
	return prediction


def test_images(images_path_1, images_path_2, model_path, cuda=True):
	if model_path is not None:
		model = torch.load(model_path)
		print('Model found in {}'.format(model_path))
	else:
		print('No model found, please check it!')
	cuda = True
	if cuda:
		model = model.cuda()
	images_list = os.listdir(images_path_1)
	real_label = [1] * len(images_list)
	real_score = []
	for images in images_list:
		image = cv2.imread(os.path.join(images_path_1, images))
		prediction = predict_with_model(image, model, cuda=cuda)
		real_score.append(prediction)


	images_list = os.listdir(images_path_2)
	fake_label = [0] * len(images_list)
	fake_score = []
	for images in images_list:
		image = cv2.imread(os.path.join(images_path_2, images))
		prediction = predict_with_model(image, model, cuda=cuda)
		fake_score.append(prediction)

	label = real_label + fake_label
	score = real_score + fake_score
	#print(label, score)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	fpr, tpr, _ = roc_curve(label, score)
	roc_auc = auc(fpr, tpr)
	print(roc_auc)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.show()


if __name__ == '__main__':
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--images_path_1', '-i1', default="extended_database/test/real", type=str)
	p.add_argument('--images_path_2', '-i2', default="extended_database/test/fake", type=str)
	p.add_argument('--model_path', '-mi', default="output/mobilenetv3_extended.pkl", type=str)
	p.add_argument('--cuda', action='store_true')
	args = p.parse_args()
	test_images(**vars(args))
