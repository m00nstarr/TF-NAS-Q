import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import time
import copy
from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenetv2 import InvertedResidual


INIT_TIMES = 100
LAT_TIMES  = 1000

def measure_act_memory(model, input_shape, is_cuda):
	
	model.eval()

	x = torch.randn(input_shape)
	if is_cuda:
		model = model.cuda()
		x = x.cuda()
	else:
		model = model.cpu()
		x = x.cpu()
	
	with torch.no_grad():
		for _ in range(10):
			output = model(x)

		output = model(x)
		mem_list = model.get_activation_memory()

		peak_mem = max(mem_list)
	
	return peak_mem


def measure_latency_in_ms(model, input_shape, is_cuda):
	lat = AverageMeter()
	model.eval()

	x = torch.randn(input_shape)
	if is_cuda:
		model = model.cuda()
		x = x.cuda()
	else:
		model = model.cpu()
		x = x.cpu()

	with torch.no_grad():
		for _ in range(INIT_TIMES):
			output = model(x)

		for _ in range(LAT_TIMES):
			tic = time.time()
			output = model(x)
			toc = time.time()
			lat.update(toc-tic, x.size(0))

	return lat.avg * 1000 # save as ms


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def drop_connect(x, training=False, drop_connect_rate=0.0):
	"""Apply drop connect."""
	if not training:
		return x
	keep_prob = 1 - drop_connect_rate
	random_tensor = keep_prob + torch.rand(
		(x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
	random_tensor.floor_()  # binarize
	output = x.div(keep_prob) * random_tensor
	return output


def channel_shuffle(x, groups):
	assert groups > 1
	batchsize, num_channels, height, width = x.size()
	assert (num_channels % groups == 0)
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize, groups, channels_per_group, height, width)
	# transpose
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.view(batchsize, -1, height, width)
	return x


def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: {}'.format(kernel_size)
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.makedirs(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)

def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32):
	act_byte = activation_bits / 8
	model = copy.deepcopy(net)

	# noinspection PyArgumentList
	def count_convNd(m, x, y):			
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes
		else:
			m.grad_activations = torch.Tensor([0]).cuda()
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups]).cuda()  # bytes

	# noinspection PyArgumentList
	def count_linear(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes
		else:	
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte]).cuda()  # bytes

	# noinspection PyArgumentList
	def count_bn(m, x, _):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes
		else:
			m.grad_activations = torch.Tensor([0]).cuda()
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes

	# noinspection PyArgumentList
	def count_relu(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() / 8]).cuda()  # bytes
		else:
			m.grad_activations = torch.Tensor([0]).cuda()
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes

	# noinspection PyArgumentList
	def count_smooth_act(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes
		else:
			m.grad_activations = torch.Tensor([0]).cuda()
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte]).cuda()  # bytes

	def add_hooks(m_):
		if len(list(m_.children())) > 0:
			return

		m_.register_buffer('grad_activations', torch.zeros(1).cuda())
		m_.register_buffer('tmp_activations', torch.zeros(1).cuda())

		if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
			fn = count_convNd
		elif type(m_) in [nn.Linear]:
			fn = count_linear
		elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
			fn = count_bn
		elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
			fn = count_relu
		elif type(m_) in [nn.Sigmoid, nn.Tanh, nn.Hardswish, nn.Hardsigmoid]:
			fn = count_smooth_act
		else:
			fn = None

		if fn is not None:
			_handler = m_.register_forward_hook(fn)

	model.eval()
	model.apply(add_hooks)

	x = torch.zeros(input_size).cuda()#.to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x, sampling = False, mode= 'max')

	memory_info_dict = {
		'peak_activation_size': torch.zeros(1).cuda(),
		'grad_activation_size': torch.zeros(1).cuda(),
		'residual_size': torch.zeros(1).cuda(),
	}

	model.cuda()
	for m in model.modules():
		if len(list(m.children())) == 0:
			def new_forward(_module):
				def lambda_forward(_x):
					current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
					                   memory_info_dict['residual_size']
					memory_info_dict['peak_activation_size'] = max(
						current_act_size, memory_info_dict['peak_activation_size']
					)
					memory_info_dict['grad_activation_size'] += _module.grad_activations
					return _module.old_forward(_x)

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		if (isinstance(m, ResidualBlock) and m.shortcut is not None) or \
				(isinstance(m, InvertedResidual) and m.use_res_connect) or \
				type(m) in [BasicBlock, Bottleneck]:
			def new_forward(_module):
				def lambda_forward(_x):
					memory_info_dict['residual_size'] = _x.numel() * act_byte
					result = _module.old_forward(_x)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

	x = torch.zeros(input_size).cuda()#.to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x, sampling = False, mode = 'max')

	return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()