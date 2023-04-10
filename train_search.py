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
from pathlib import Path
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
from tools.config import mc_mask_dddict, peak_memory_lookup_key_dddict
from models.model_search import Network
from parsing_model import get_op_and_depth_quantize_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import ImageList, IMAGENET_MEAN, IMAGENET_STD



parser = argparse.ArgumentParser("searching TF-NAS")
# various path
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
parser.add_argument('--num_classes', type=int, default=200, help='class number of training set')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# hyper parameters
parser.add_argument('--target_memory', type=float, default = 0.5, help = 'the target activation memory')
parser.add_argument('--peak_memory_lookup_path', type=str, default = "./latency_pkl/peak_memory_cpu.pkl", help = 'path of memory lookup table')
parser.add_argument('--cuda_device', type=str, default = "0", help = "gpu_id")


args = parser.parse_args()

if args.save == "./checkpoints":
	args.save = os.path.join(args.save, 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
	print(f'new exp dir: {args.save}')


create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

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

	with open(args.peak_memory_lookup_path, 'rb') as f:
		peak_memory_lookup = pickle.load(f)

	mc_maxnum_dddict = get_mc_num_dddict(mc_mask_dddict, is_max=True)
	model = Network(args.num_classes, mc_maxnum_dddict, target_memory=args.target_memory, peak_memory_lookup=peak_memory_lookup)
	model = torch.nn.DataParallel(model).cuda()
	model.module.set_temperature(args.T)
	logging.info("param size = %fMB", count_parameters_in_MB(model))

	# save initial model
	model_path = Path(os.path.join(args.save, 'searched_model_00.pth.tar'))
	torch.save({
			'state_dict': model.state_dict(),
			'mc_mask_dddict': mc_mask_dddict,
		}, model_path)

	# get lr list
	lr_w_list = []
	lr_g_list = []

	optimizer_w = torch.optim.SGD(
					model.module.weight_parameters(),
					lr = args.w_lr,
					momentum = args.w_mom,
					weight_decay = args.w_wd)

	scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, float(args.epochs))

	optimizer_g = torch.optim.SGD(
						model.module.arch_parameters(),
						lr = args.a_lr,
						momentum = args.w_mom,
						weight_decay = args.a_wd)

	scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, float(args.epochs))
	
	for _ in range(args.epochs):
		lr_w = scheduler_w.get_lr()[0]
		lr_w_list.append(lr_w)
		scheduler_w.step()
		lr_g = scheduler_g.get_lr()[0]
		lr_g_list.append(lr_g)
		scheduler_g.step()


	del model
	del optimizer_w
	del scheduler_w
	del optimizer_g
	del scheduler_g

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
	train_transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
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
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])

	train_queue = torch.utils.data.DataLoader(
		ImageList(list_path=args.train_list, transform=train_transform), 
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

	val_queue = torch.utils.data.DataLoader(
		ImageList(list_path=args.val_list, transform=val_transform), 
		batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)


	for epoch in range(args.epochs):

		mc_num_dddict = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, mc_num_dddict, target_memory=args.target_memory, peak_memory_lookup=peak_memory_lookup)
		model = torch.nn.DataParallel(model).cuda()
		model.module.set_temperature(args.T)

		# load model
		prev_model_path = Path(os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch)))
		next_model_path = Path(os.path.join(args.save, 'searched_model_{:02}.pth.tar'.format(epoch+1)))

		if next_model_path.exists():
			logging.info('Epoch: %d - already done! pass...', epoch)
			continue

		state_dict = torch.load(prev_model_path)['state_dict']

		for key in state_dict:
			if 'm_ops' not in key:
				exec('model.{}.data = state_dict[key].data'.format(key))

		for stage in mc_mask_dddict:
			for block in mc_mask_dddict[stage]:
				for op_idx in mc_mask_dddict[stage][block]:
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1).cuda()
					iw = 'model.module.{}.{}.m_ops[{}].inverted_bottleneck.conv.weight.data'.format(stage, block, op_idx)
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					exec(iw + ' = torch.index_select(state_dict[iw_key], 0, index).data')
					dw = 'model.module.{}.{}.m_ops[{}].depth_conv.conv.weight.data'.format(stage, block, op_idx)
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					exec(dw + ' = torch.index_select(state_dict[dw_key], 0, index).data')
					pw = 'model.module.{}.{}.m_ops[{}].point_linear.conv.weight.data'.format(stage, block, op_idx)
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					exec(pw + ' = torch.index_select(state_dict[pw_key], 1, index).data')


		del index

		lr_w = lr_w_list[epoch]
		lr_g = lr_g_list[epoch]

		optimizer_w = torch.optim.SGD(
						model.module.weight_parameters(),
						lr = lr_w,
						momentum = args.w_mom,
						weight_decay = args.w_wd)
		optimizer_a = torch.optim.Adam(
						model.module.arch_parameters(),
						lr = args.a_lr,
						betas = (args.a_beta1, args.a_beta2),
						weight_decay = args.a_wd)
		optimizer_g = torch.optim.SGD(
						model.module.quantized_parameters(),
						lr = lr_g,
						momentum = args.w_mom,
						weight_decay = args.a_wd)

		logging.info('Epoch: %d lr_w: %e  lr_g : %e T: %e', epoch, lr_w, lr_g , args.T)

		# training
		epoch_start = time.time()
		if epoch < 10:
			train_acc = train_wo_arch(train_queue, model, criterion, optimizer_w)
			# 이 아래로는 가지도 못함 애초에.
		else:
			train_acc = train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a, optimizer_g)
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

		# logging gamma (quant) params
		stage =[1,1,2,2,2,3,3,3,3,4,4,4,4,5]
		block= [1,2,1,2,3,1,2,3,4,1,2,3,4,1]
		idx = 0
		for param in model.module.gammas_parameters():
			param = F.softmax(param.detach().cpu(), dim=-1)
			param = param.numpy()
			logging.info('stage: ' + str(stage[idx])+ ' block: ' + str(block[idx])+ ' /' +' '.join(['{:.6f}'.format(p) for p in param]))
			idx += 1

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
					index = torch.nonzero(mc_mask_dddict[stage][block][op_idx]).view(-1).cuda()
					iw_key = 'module.{}.{}.m_ops.{}.inverted_bottleneck.conv.weight'.format(stage, block, op_idx)
					state_dict[iw_key].data[index,:,:,:] = state_dict_from_model[iw_key]
					dw_key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
					state_dict[dw_key].data[index,:,:,:] = state_dict_from_model[dw_key]
					pw_key = 'module.{}.{}.m_ops.{}.point_linear.conv.weight'.format(stage, block, op_idx)
					state_dict[pw_key].data[:,index,:,:] = state_dict_from_model[pw_key]
				
		del state_dict_from_model, index

		# shrink and expand
		if epoch >= 10:
			logging.info('Now shrinking or expanding the arch')
			op_weights, depth_weights, quantize_weights = get_op_and_depth_quantize_weights(model)

			parsed_arch = parse_architecture(op_weights, depth_weights, quantize_weights)
			mc_num_dddict = get_mc_num_dddict(mc_mask_dddict)

			# moon: stage마다 memory activation을 구하는 반면 latency table이 통째로 재가지고 그렇다(?)

			for cur_stage in range(1, 6):
				stage = 'stage{}'.format(cur_stage)
				cur_before_mem = get_lookup_peak_memory(parsed_arch, mc_num_dddict, peak_memory_lookup_key_dddict, peak_memory_lookup, stage)
				logging.info('Before, the current memory of {}: {:.4f}, the target peak memory: {:.4f}'.format(stage, cur_before_mem, args.target_memory))

				if cur_before_mem > args.target_memory:
					logging.info('Stage{} Shrinking.....'.format(cur_stage))
					mc_num_dddict, cur_after_mem = fit_mc_num_by_peak_memory(parsed_arch, mc_num_dddict, mc_maxnum_dddict, 
															peak_memory_lookup_key_dddict, peak_memory_lookup, args.target_memory, stage, sign=-1)
				elif cur_before_mem < args.target_memory:
					logging.info('Stage{} Expanding.....'.format(cur_stage))
					mc_num_dddict, cur_after_mem = fit_mc_num_by_peak_memory(parsed_arch, mc_num_dddict, mc_maxnum_dddict, 
															peak_memory_lookup_key_dddict, peak_memory_lookup, args.target_memory, stage, sign=1)
				else:
					logging.info('Stage{} No Operation...'.format(cur_stage))
					cur_after_mem = cur_before_mem

				logging.info('After, the {} current peakmemory: {:.4f}, the target peakmemory: {:.4f}'.format(stage, cur_after_mem, args.target_memory))
		

			# change mc_mask_dddict based on mc_num_dddict
			for stage in parsed_arch:
				for block in parsed_arch[stage]:
					op_idx = parsed_arch[stage][block][0]
					if mc_num_dddict[stage][block][op_idx] != int(sum(mc_mask_dddict[stage][block][op_idx]).item()):
						mc_num = mc_num_dddict[stage][block][op_idx]
						max_mc_num = mc_mask_dddict[stage][block][op_idx].size(0)
						mc_mask_dddict[stage][block][op_idx].data[[True]*max_mc_num] = 0.0
						key = 'module.{}.{}.m_ops.{}.depth_conv.conv.weight'.format(stage, block, op_idx)
						weight_copy = state_dict[key].clone().abs().cpu().numpy()
						weight_l1_norm = np.sum(weight_copy, axis=(1,2,3))
						weight_l1_order = np.argsort(weight_l1_norm)
						weight_l1_order_rev = weight_l1_order[::-1][:mc_num]
						mc_mask_dddict[stage][block][op_idx].data[weight_l1_order_rev.tolist()] = 1.0

		# save model
		torch.save({
				'state_dict': state_dict,
				'mc_mask_dddict': mc_mask_dddict,
			}, next_model_path)


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
		# https://discuss.pytorch.org/t/error-host-softmax-not-implemented-for-long/74951
		
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

def train_w_arch(train_queue, val_queue, model, criterion, optimizer_w, optimizer_a, optimizer_g):
	objs_a = AverageMeter()
	objs_w = AverageMeter()
	objs_g = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	model.train()

	for step, (x_w, target_w) in enumerate(train_queue):
		x_w = x_w.cuda(non_blocking=True)
		target_w = target_w.cuda(non_blocking=True)

		for param in model.module.weight_parameters():
			param.requires_grad = True
		for param in model.module.arch_parameters():
			param.requires_grad = False
		for param in model.module.quantized_parameters():
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

			x_a = x_a.cuda(non_blocking=True)
			target_a = target_a.cuda(non_blocking=True)

			for param in model.module.weight_parameters():
				param.requires_grad = False
			for param in model.module.arch_parameters():
				param.requires_grad = True
			for param in model.module.quantized_parameters():
				param.requires_grad = False		


			logits_a, loss_peak_memory = model(x_a, sampling=False)
			loss_a = criterion(logits_a, target_a)

			optimizer_a.zero_grad()
			loss_a.backward()
			if args.grad_clip > 0:
				nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)

			optimizer_a.step()

			# ensure log_alphas to be a log probability distribution
			for log_alphas in model.module.arch_parameters():
				log_alphas.data = F.log_softmax(log_alphas.detach().data, dim=-1)
			
			#quantization loss 함수 정의 후 추가하였음
			#loss_q -> 모든 레이어 별 (peak memory와 target memory의 차이)의 총합
			#gamma에 대해 학습하기

			for param in model.module.weight_parameters():
				param.requires_grad = False
			for param in model.module.arch_parameters():
				param.requires_grad = False
			for param in model.module.quantized_parameters():
				param.requires_grad = True

			logits_a, loss_peak_memory = model(x_a, sampling=False)
			loss_q = loss_peak_memory

			optimizer_g.zero_grad()
			loss_q.backward()
			if args.grad_clip > 0:
				nn.utils.clip_grad_norm_(model.module.quantized_parameters(), args.grad_clip)
			optimizer_g.step()

			n = x_a.size(0)
			objs_a.update(loss_a.item(), n)
			objs_g.update(loss_q.item(), n)
		
		if step % args.print_freq == 0:
			logging.info('TRAIN w_Arch Step: %04d Objs_W: %f R1: %f R5: %f Objs_A: %f loss_q: %f', 
						  step, objs_w.avg, top1.avg, top5.avg, objs_a.avg, objs_g.avg)

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
			logits, _ = model(x, sampling=True, mode='gumbel')
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


#stage단위의 peak memory 를 줌..
def get_lookup_peak_memory(parsed_arch, mc_num_dddict, peak_memory_lookup_key_dddict, peak_memory_lookup, stage):
	peak_memory = 0.
	
	for block in parsed_arch[stage]:
		op_idx = parsed_arch[stage][block][0]
		quant_idx = parsed_arch[stage][block][1]
		mid_channels_key = mc_num_dddict[stage][block][op_idx]
		peak_memory_lookup_key = peak_memory_lookup_key_dddict[stage][block][op_idx]
		cur_peak_memory = peak_memory_lookup[peak_memory_lookup_key][mid_channels_key]
		# -> 0: 8bit, 1: 16bit, 2: 32bit가 되게끔 수정해야 함.
		cur_peak_memory = cur_peak_memory / (2**(2-quant_idx))
		peak_memory = max(peak_memory, cur_peak_memory)

	return peak_memory



def fit_mc_num_by_peak_memory(parsed_arch, mc_num_dddict, mc_maxnum_dddict, peak_memory_lookup_key_dddict, peak_memory_lookup, target_memory, stage, sign):
	# sign=1 for expand / sign=-1 for shrink
	assert sign == -1 or sign == 1
	peak_memory = get_lookup_peak_memory(parsed_arch, mc_num_dddict, peak_memory_lookup_key_dddict, peak_memory_lookup, stage)

	parsed_mc_num_list = []
	parsed_mc_maxnum_list = []
	for block in parsed_arch[stage]:
		op_idx = parsed_arch[stage][block][0]
		parsed_mc_num_list.append(mc_num_dddict[stage][block][op_idx])
		parsed_mc_maxnum_list.append(mc_maxnum_dddict[stage][block][op_idx])

	new_mc_num_dddict = copy.deepcopy(mc_num_dddict)
	new_peak_memory = peak_memory
	parsed_mc_bound_switches = [False]*len(parsed_mc_num_list) # 각 block이 bound 넘었는지 아닌지 저장 (false: bound 안넘음, true: bound 넘음)

	while (sign*new_peak_memory <= sign*target_memory):
		mc_num_dddict = copy.deepcopy(new_mc_num_dddict)
		peak_memory = new_peak_memory
		list_idx = 0
		
		for block in parsed_arch[stage]:
			# 아직 범위가 안 넘어간 block만 mc 수 값 변경
			if parsed_mc_bound_switches[list_idx]==False:
				op_idx = parsed_arch[stage][block][0]
				max_mc = parsed_mc_maxnum_list[list_idx]
				min_mc = max_mc // 2
				new_mc_num = mc_num_dddict[stage][block][op_idx] + sign
				#범위 넘어가거나 최소값이 되면 parsed_mc_bound_switches 에 false 값 넣기
				if new_mc_num > max_mc or new_mc_num < min_mc:
					parsed_mc_bound_switches[list_idx] = True
					print(new_mc_num, ': exceed the bound!')
					print(parsed_mc_bound_switches)
				else:
					new_mc_num_dddict[stage][block][op_idx] = new_mc_num
			list_idx += 1

		if all(parsed_mc_bound_switches):
			break

		new_peak_memory = get_lookup_peak_memory(parsed_arch, new_mc_num_dddict, peak_memory_lookup_key_dddict, peak_memory_lookup, stage)


	if sign == -1:
		mc_num_dddict = copy.deepcopy(new_mc_num_dddict)
		peak_memory = new_peak_memory

	return mc_num_dddict, peak_memory


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
