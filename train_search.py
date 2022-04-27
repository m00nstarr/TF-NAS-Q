import os
import sys
import time
import glob
import logging
import argparse
import pickle
import copy
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision

from tools.utils import AverageMeter, accuracy
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir
from tools.config import mc_mask_dddict, lat_lookup_key_dddict
from models.model_search import Network
from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import ImageList, IMAGENET_MEAN, IMAGENET_STD
from tools.utils import count_activation_size

from collections import OrderedDict

parser = argparse.ArgumentParser("searching TF-NAS")
# various path
parser.add_argument('--img_root', type=str, required=True, help='image root path (ImageNet train set)')
parser.add_argument('--train_list', type=str, default="./dataset/ImageNet-100-effb0_train_cls_ratio0.8.txt",
					help='training image list')
parser.add_argument('--val_list', type=str, default="./dataset/ImageNet-100-effb0_val_cls_ratio0.8.txt",
					help='validating image list')
parser.add_argument('--lookup_path', type=str, default="./latency_pkl/latency_gpu.pkl",
					help='path of lookup table')
parser.add_argument('--save', type=str, default='./checkpoints', help='model and log saving path')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=90, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--w_lr', type=float, default=0.025, help='learning rate for weights')
parser.add_argument('--w_mom', type=float, default=0.9, help='momentum for weights')
parser.add_argument('--w_wd', type=float, default=1e-5, help='weight decay for weights')
parser.add_argument('--a_lr', type=float, default=0.01, help='learning rate for arch')
parser.add_argument('--a_wd', type=float, default=5e-4, help='weight decay for arch')
parser.add_argument('--a_beta1', type=float, default=0.5, help='beta1 for arch')
parser.add_argument('--a_beta2', type=float, default=0.999, help='beta2 for arch')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--T', type=float, default=5.0, help='temperature for gumbel softmax')
parser.add_argument('--T_decay', type=float, default=0.96, help='temperature decay')
parser.add_argument('--num_classes', type=int, default=100, help='class number of training set')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# hyper parameters
parser.add_argument('--lambda_lat', type=float, default=0.1, help='trade off for latency')
parser.add_argument('--target_lat', type=float, default=15.0, help='the target latency')
parser.add_argument('--target_memory', type=float, default = 0.5, help = 'the target activation memory')


args = parser.parse_args()

args.save = os.path.join(args.save, 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
#args.save = os.path.join(args.save,'search-20220413-162213-'+args.note)
#args.save = os.path.join(args.save,'search-20220414-141341-'+args.note)
#args.save = os.path.join(args.save,'search-20220418-111514-'+args.note)
#args.save = os.path.join(args.save,'search-20220421-163232-'+args.note)
create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


activation_memory_dict = OrderedDict([
		('stage1', OrderedDict([
				('block1', OrderedDict([
						(0, 1.515625),
						(1, 3.015625),
						(2, 1.515625),
						(3, 3.015625),
					])),
				('block2', OrderedDict([
						(0, 0.568359),
						(1, 1.130859),
						(2, 0.568359),
						(3, 1.130859),
					])),
			])),
		('stage2', OrderedDict([
				('block1', OrderedDict([
						(0, 0.568359),
						(1, 1.130859),
						(2, 0.568359),
						(3, 1.130859),
					])),
				('block2', OrderedDict([
						(0, 0.236816),
						(1, 0.471191),
						(2, 0.236816),
						(3, 0.471191),
					])),
				('block3', OrderedDict([
						(0, 0.236816),
						(1, 0.471191),
						(2, 0.236816),
						(3, 0.471191),
					])),
			])),
		('stage3', OrderedDict([
				('block1', OrderedDict([
						(0, 0.236816),
						(1, 0.471191),
						(2, 0.236816),
						(3, 0.471191),
					])),
				('block2', OrderedDict([
						(0, 0.118408),
						(1, 0.235596),
						(2, 0.118408),
						(3, 0.235596),
					])),
				('block3', OrderedDict([
						(0, 0.118408),
						(1, 0.235596),
						(2, 0.118408),
						(3, 0.235596),
					])),
				('block4', OrderedDict([
						(0, 0.118408),
						(1, 0.235596),
						(2, 0.118408),
						(3, 0.235596),
					])),
			])),
		('stage4', OrderedDict([
				('block1', OrderedDict([
						(0, 0.118408),
						(1, 0.235596),
						(2, 0.118408),
						(3, 0.235596),
					])),
				('block2', OrderedDict([
						(0, 0.165771),
						(1, 0.329834),
						(2, 0.165771),
						(3, 0.329834),
					])),
				('block3', OrderedDict([
						(0, 0.165771),
						(1, 0.329834),
						(2, 0.165771),
						(3, 0.329834),
					])),
				('block4', OrderedDict([
						(0, 0.165771),
						(1, 0.329834),
						(2, 0.165771),
						(3, 0.329834),
					])),
			])),
		('stage5', OrderedDict([
				('block1', OrderedDict([
						(0, 0.165771),
						(1, 0.329834),
						(2, 0.165771),
						(3, 0.329834),
					])),
			])),
	])

def main():
	if not torch.cuda.is_available():
		logging.info('No GPU device available')
		sys.exit(1)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	cudnn.enabled=True
	cudnn.benchmark = True
	logging.info("args = %s", args)

	
	mc_maxnum_dddict = get_mc_num_dddict(mc_mask_dddict, is_max=True)
	model = Network(args.num_classes, mc_maxnum_dddict)
	model = torch.nn.DataParallel(model).cuda()
	model.module.set_temperature(args.T)
	logging.info("param size = %fMB", count_parameters_in_MB(model))

	# save initial model
	model_path = os.path.join(args.save, 'searched_model_00.pth.tar')
	torch.save({
			'state_dict': model.state_dict(),
			'mc_mask_dddict': mc_mask_dddict,
		}, model_path)

	# get lr list
	lr_list = []
	optimizer_w = torch.optim.SGD(
					model.module.weight_parameters(),
					lr = args.w_lr,
					momentum = args.w_mom,
					weight_decay = args.w_wd)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, float(args.epochs))
	for _ in range(args.epochs):
		lr = scheduler.get_lr()[0]
		lr_list.append(lr)
		scheduler.step()
	del model
	del optimizer_w
	del scheduler

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
	train_transform = transforms.Compose([
			#transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(
				brightness=0.4,
				contrast=0.4,
				saturation=0.4,
				hue=0.2),
			transforms.ToTensor(),
			normalize,
		])

	val_transform = transforms.Compose([
			#transforms.Resize(256),
			# transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])

	# train_queue = torch.utils.data.DataLoader(
	# 	ImageList(root=args.img_root, 
	# 			  list_path=args.train_list, 
	# 			  transform=train_transform), 
	# 	batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

	# val_queue = torch.utils.data.DataLoader(
	# 	ImageList(root=args.img_root, 
	# 			  list_path=args.val_list, 
	# 			  transform=val_transform), 
	# 	batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
	
	# cifar-10
	train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
	train_queue = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                 shuffle=True, pin_memory=True, num_workers=args.workers)

	val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=val_transform)
	val_queue = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=True, num_workers=args.workers)

	for epoch in range(10,args.epochs):
		
		mc_num_dddict = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, mc_num_dddict)
		model = torch.nn.DataParallel(model).cuda()
		model.module.set_temperature(args.T)

		# load model
		model_path = os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch))
		state_dict = torch.load(model_path)['state_dict']
		for key in state_dict:
			if 'm_ops' not in key:
				exec('model.{}.data = state_dict[key].data'.format(key))
		for stage in mc_mask_dddict:
			for block in mc_mask_dddict[stage]:
				for op_idx in mc_mask_dddict[stage][block]:
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1)
					index = index.cuda()
					iw = 'model.module.{}.{}.m_ops[{}].inverted_bottleneck.conv.weight.data'.format(stage, block, op_idx)
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					exec(iw + ' = torch.index_select(state_dict[iw_key], 0, index).data')
					dw = 'model.module.{}.{}.m_ops[{}].depth_conv.conv.weight.data'.format(stage, block, op_idx)
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					exec(dw + ' = torch.index_select(state_dict[dw_key], 0, index).data')
					pw = 'model.module.{}.{}.m_ops[{}].point_linear.conv.weight.data'.format(stage, block, op_idx)
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					exec(pw + ' = torch.index_select(state_dict[pw_key], 1, index).data')

					if op_idx >= 4:
						se_cr_w = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_reduce.weight.data'.format(stage, block, op_idx)
						se_cr_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.weight'.format(stage, block, op_idx)
						exec(se_cr_w + ' = torch.index_select(state_dict[se_cr_w_key], 1, index).data')
						se_cr_b = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_reduce.bias.data'.format(stage, block, op_idx)
						se_cr_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.bias'.format(stage, block, op_idx)
						exec(se_cr_b + ' = state_dict[se_cr_b_key].data')
						se_ce_w = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_expand.weight.data'.format(stage, block, op_idx)
						se_ce_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.weight'.format(stage, block, op_idx)
						exec(se_ce_w + ' = torch.index_select(state_dict[se_ce_w_key], 0, index).data')
						se_ce_b = 'model.module.{}.{}.m_ops[{}].squeeze_excite.conv_expand.bias.data'.format(stage, block, op_idx)
						se_ce_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.bias'.format(stage, block, op_idx)
						exec(se_ce_b + ' = torch.index_select(state_dict[se_ce_b_key], 0, index).data')
		del index

		lr = lr_list[epoch]
		optimizer_w = torch.optim.SGD(
						model.module.weight_parameters(),
						lr = lr,
						momentum = args.w_mom,
						weight_decay = args.w_wd)
		optimizer_a = torch.optim.Adam(
						model.module.arch_parameters(),
						lr = args.a_lr,
						betas = (args.a_beta1, args.a_beta2),
						weight_decay = args.a_wd)
		logging.info('Epoch: %d lr: %e T: %e', epoch, lr, args.T)
		#logging.info("peak activation memory size = %fMB", count_activation_size(model)[0]/1048576.0)
		
		op_weights, depth_weights, quantized_weights = get_op_and_depth_weights(model)
		
		# training
		epoch_start = time.time()
		if epoch < 10:
			train_acc = train_wo_arch(train_queue, model, criterion, optimizer_w)
		else:
			train_acc = train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a)
			args.T *= args.T_decay
		
		# logging arch parameters
		logging.info('The current arch parameters are:')
		for param in model.module.log_alphas_parameters():
			param = np.exp(param.detach().cpu().numpy())
			logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
		
		for param in model.module.betas_parameters():
			param = F.softmax(param.detach().cpu(), dim=-1)
			param = param.numpy()
			logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
		
		for param in model.module.gammas_parameters():
			param = F.softmax(param.detach().cpu(), dim=-1)
			param = param.numpy()
			logging.info(' '.join(['{:.6f}'.format(p) for p in param]))
		
		logging.info('Train_acc %f', train_acc)
		epoch_duration = time.time() - epoch_start
		logging.info('Epoch time: %ds', epoch_duration)

		# validation for last 5 epochs
		if args.epochs - epoch < 5:
			val_acc = validate(val_queue, model, criterion)
			logging.info('Val_acc %f', val_acc)

		# update state_dict
		state_dict_from_model = model.state_dict()
		for key in state_dict:
			if 'm_ops' not in key:
				state_dict[key].data = state_dict_from_model[key].data
		for stage in mc_mask_dddict:
			for block in mc_mask_dddict[stage]:
				for op_idx in mc_mask_dddict[stage][block]:
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1)
					index = index.cuda()
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					state_dict[iw_key].data[index,:,:,:] = state_dict_from_model[iw_key]
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					state_dict[dw_key].data[index,:,:,:] = state_dict_from_model[dw_key]
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					state_dict[pw_key].data[:,index,:,:] = state_dict_from_model[pw_key]
					if op_idx >= 4:
						se_cr_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.weight'.format(stage, block, op_idx)
						state_dict[se_cr_w_key].data[:,index,:,:] = state_dict_from_model[se_cr_w_key]
						se_cr_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_reduce.bias'.format(stage, block, op_idx)
						state_dict[se_cr_b_key].data[:] = state_dict_from_model[se_cr_b_key]
						se_ce_w_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.weight'.format(stage, block, op_idx)
						state_dict[se_ce_w_key].data[index,:,:,:] = state_dict_from_model[se_ce_w_key]
						se_ce_b_key = 'module.{}.{}.m_ops.{}.squeeze_excite.conv_expand.bias'.format(stage, block, op_idx)
						state_dict[se_ce_b_key].data[index] = state_dict_from_model[se_ce_b_key]
		del state_dict_from_model, index

		# save model
		model_path = os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch+1))
		torch.save({
				'state_dict': state_dict,
				'mc_mask_dddict': mc_mask_dddict,
			}, model_path)


def train_wo_arch(train_queue, model, criterion, optimizer_w):
	objs_w = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	model.train()

	for param in model.module.weight_parameters():
		param.requires_grad = True
	for param in model.module.arch_parameters():
		param.requires_grad = False

	for step, (x_w, target_w) in enumerate(train_queue):
		x_w = x_w.cuda(non_blocking=True)
		target_w = target_w.cuda(non_blocking=True)

		logits_w_gumbel, _ = model(x_w, sampling=True, mode='gumbel')
		loss_w_gumbel = criterion(logits_w_gumbel, target_w)
		# reset switches of log_alphas
		model.module.reset_switches()

		optimizer_w.zero_grad()
		loss_w_gumbel.backward()
		if args.grad_clip > 0:
			nn.utils.clip_grad_norm_(model.module.weight_parameters(), args.grad_clip)
		optimizer_w.step()

		prec1, prec5 = accuracy(logits_w_gumbel, target_w, topk=(1, 5))
		n = x_w.size(0)
		objs_w.update(loss_w_gumbel.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % args.print_freq == 0:
			logging.info('TRAIN wo_Arch Step: %04d Objs: %f R1: %f R5: %f', step, objs_w.avg, top1.avg, top5.avg)

	return top1.avg


def train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a):
	objs_a = AverageMeter()
	objs_w = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	model.train()

	for step, (x_w, target_w) in enumerate(train_queue):
		x_w = x_w.cuda()#(non_blocking=True)
		target_w = target_w.cuda()#(non_blocking=True)

		for param in model.module.weight_parameters():
			param.requires_grad = True
		for param in model.module.arch_parameters():
			param.requires_grad = False
		
		logits_w_gumbel, _ = model(x_w, sampling=True, mode='gumbel')
		loss_w_gumbel = criterion(logits_w_gumbel, target_w)
		logits_w_random, _ = model(x_w, sampling=True, mode='random')
		loss_w_random = criterion(logits_w_random, target_w)
		loss_w = loss_w_gumbel + loss_w_random
		
		optimizer_w.zero_grad()
		loss_w.backward()
		if args.grad_clip > 0:
			nn.utils.clip_grad_norm_(model.module.weight_parameters(), args.grad_clip)
		optimizer_w.step()

		prec1, prec5 = accuracy(logits_w_gumbel, target_w, topk=(1, 5))
		n = x_w.size(0)
		objs_w.update(loss_w.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % 2 == 0:
			# optimize a
			try:
				x_a, target_a = next(val_queue_iter)
			except:
				val_queue_iter = iter(val_queue)
				x_a, target_a = next(val_queue_iter)

			x_a = x_a.cuda()#non_blocking=True)
			target_a = target_a.cuda()#non_blocking=True)

			for param in model.module.weight_parameters():
				param.requires_grad = False
			for param in model.module.arch_parameters():
				param.requires_grad = True

			logits_a, peak_memory = model(x_a, sampling=False)
			loss_a = criterion(logits_a, target_a)
			
			#quantization loss 함수 정의 후 추가하였음
			#peak memory 와 taget.memory (0.5MB)의 차이를 loss_q값 정의 
			#
			loss_q = abs(peak_memory / args.target_memory - 1.) * 0.4

			loss = loss_a + loss_q

			optimizer_a.zero_grad()
			loss.backward()
			if args.grad_clip > 0:
				nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
			optimizer_a.step()

			# ensure log_alphas to be a log probability distribution
			for log_alphas in model.module.arch_parameters():
				log_alphas.data = F.log_softmax(log_alphas.detach().data, dim=-1)
			
			n = x_a.size(0)
			objs_a.update(loss_a.item(), n)


		if step % args.print_freq == 0:
			logging.info('TRAIN w_Arch Step: %04d Objs_W: %f R1: %f R5: %f Objs_A: %f loss_q: %f' , 
						  step, objs_w.avg, top1.avg, top5.avg, objs_a.avg, loss_q)

	return top1.avg


def validate(val_queue, model, criterion):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# model.eval()
	# disable moving average
	model.train()

	for step, (x, target) in enumerate(val_queue):
		x = x.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)
		with torch.no_grad():
			logits, _, _ = model(x, sampling=True, mode='gumbel')
			loss = criterion(logits, target)
		# reset switches of log_alphas
		model.module.reset_switches()

		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		n = x.size(0)
		objs.update(loss.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % args.print_freq == 0:
			logging.info('VALIDATE Step: %04d Objs: %f R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

	return top1.avg

def bound_clip(mc_num, max_mc_num):
	min_mc_num = max_mc_num // 2

	if mc_num <= min_mc_num:
		new_mc_num = min_mc_num
		switch = False
	elif mc_num >= max_mc_num:
		new_mc_num = max_mc_num
		switch = False
	else:
		new_mc_num = mc_num
		switch = True

	return new_mc_num, switch


if __name__ == '__main__':
	start_time = time.time()
	main() 
	end_time = time.time()
	duration = end_time - start_time
	logging.info('Total searching time: %ds', duration)