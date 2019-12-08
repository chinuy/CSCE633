from __future__ import print_function, division

import torch
import torch.nn as nn
import argparse
import time
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
	cuda = True
	if cuda:
		preprocessed_image = preprocessed_image.cuda()
	return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
	cuda = True
	preprocessed_image = preprocess_image(image, cuda)
	output = model(preprocessed_image)
	output = post_function(output)
	#print(output)
	_, prediction = torch.max(output, 1)
	prediction = float(prediction.cpu().numpy())
	return int(prediction), output


def test_images(images_path_1, images_path_2, model_path, cuda=True):
	if model_path is not None:
		model = torch.load(model_path, map_location='cpu')
		print('Model found in {}'.format(model_path))
	else:
		print('No model found, please check it!')
	cuda = True
	if cuda:
		model = model.cuda()
	fake_count = 0
	real_count = 0
	images_list = os.listdir(images_path_1)
	for images in images_list:
		image = cv2.imread(os.path.join(images_path_1, images))
		prediction, output = predict_with_model(image, model, cuda=cuda)
		#print(prediction, output)
		if prediction == 0:
			fake_count += 1
		else:
			real_count += 1
	print("Testing real images: ")
	print("fake frame is:", fake_count)
	print("real frame is:", real_count)



	fake_count = 0
	real_count = 0
	images_list = os.listdir(images_path_2)
	for images in images_list:
		image = cv2.imread(os.path.join(images_path_2, images))
		prediction, output = predict_with_model(image, model, cuda=cuda)
		#print(prediction, output)
		if prediction == 0:
			fake_count += 1
		else:
			real_count += 1
	print("Testing fake images: ")
	print("fake frame is:", fake_count)
	print("real frame is:", real_count)



if __name__ == '__main__':
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--images_path_1', '-i1', default="extended_database/test/real", type=str)
	p.add_argument('--images_path_2', '-i2', default="extended_database/test/fake", type=str)
	p.add_argument('--model_path', '-mi', default="output/mobilenetv3_drop_4_layer.pkl", type=str)
	args = p.parse_args()
	test_images(**vars(args))
