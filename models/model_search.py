import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

from .quantizers import AsymmetricQuantizer
from .range_trackers import *
from tools.utils import count_activation_size

# import logging

# save = '/home/moon/tinyML/TF-NAS-Q/checkpoints/search-20220421-163232-TF-NAS-lam0.1-lat15.0-gpu'
# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
# 	format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

PRIMITIVES = [
	'MBI_k3_e3',
	'MBI_k3_e6',
	'MBI_k5_e3',
	'MBI_k5_e6',
	# 'MBI_k3_e3_se',
	# 'MBI_k3_e6_se',
	# 'MBI_k5_e3_se',
	# 'MBI_k5_e6_se',
	# 'skip',
]

OPS = {
	'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	# 'MBI_k3_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
	# 'MBI_k3_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
	# 'MBI_k5_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
	# 'MBI_k5_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
	# 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}

QUANTIZE_OPS = {
	4,
	8,
	16,
	32,
}


class MixedOP(nn.Module):
	def __init__(self, in_channels, out_channels, stride, affine, act_func, num_ops, mc_num_dict, peakmemory_list, stage_num, target_mem, memory_lookup):
		super(MixedOP, self).__init__()
		self.num_ops = num_ops
		# self.lat_lookup = lat_lookup
		self.mc_num_dict = mc_num_dict
		self.m_ops = nn.ModuleList()
		self.peakmemory_list = peakmemory_list
		self.stage_num = stage_num
		self.target_mem = target_mem
		self.memory_lookup = memory_lookup

		# for quantization
		self.activation_quantizer_16b = AsymmetricQuantizer(
			bits_precision=16,
            range_tracker=AveragedRangeTracker((1, 1, 1, 1))
		)
		
		self.activation_quantizer_8b = AsymmetricQuantizer(
            bits_precision=8,
            range_tracker=AveragedRangeTracker((1, 1, 1, 1))
        )

		self.activation_quantizer_4b = AsymmetricQuantizer(
            bits_precision=4,
            range_tracker=AveragedRangeTracker((1, 1, 1, 1))
        )

		for i in range(num_ops):
			primitive = PRIMITIVES[i]
			mid_channels = self.mc_num_dict[i]
			op = OPS[primitive](in_channels, mid_channels, out_channels, stride, affine, act_func)
			self.m_ops.append(op)

		self._initialize_log_alphas()
		self._initialize_gammas()
		self.reset_switches()


	def fink_ori_idx(self, idx):
		count = 0
		for ori_idx in range(len(self.switches)):
			if self.switches[ori_idx]:
				count += 1
				if count == (idx + 1):
					break
		return ori_idx

	def forward(self, x, sampling, mode):
		if sampling:
			weights = self.log_alphas[self.switches]
			if mode == 'gumbel':
				weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
				
				#idx 여기에서 제일 probability가 높은(log_alphas값이 큰) ops가 선택된다. 
				idx = torch.argmax(weights).item()

				#random case의 경우 gumbel로 선택된 ops 외에 선택해야하기 때문에 switch를 꺼줌
				self.switches[idx] = False
			elif mode == 'gumbel_2':
				weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
				idx = torch.argmax(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'min_alphas':
				idx = torch.argmin(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'max_alphas':
				idx = torch.argmax(weights).item()
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			elif mode == 'random':
				# gumbel로 선택된것 외에 random으로 ops선택.
				idx = random.choice(range(len(weights)))
				idx = self.fink_ori_idx(idx)
				self.reset_switches()
			else:
				raise ValueError('invalid sampling mode...')
			
			#이 block에서는 위 sampling 과정을 통해 얻은 idx값을 operation으로 선택한다.
			op = self.m_ops[idx]

			return op(x), 0

		else:
			weights = F.gumbel_softmax(self.log_alphas, self.T, hard=False)
			g_weights = F.softmax(self.gammas, dim = -1)

			#peak_mem 또한 반영
			#4개의 oepration 별 peak memory 체크

			peak_mem = self.get_lookup_memory(x.size(-1))
			#4개의 OP average peak memory
			average_peak_mem = sum(w * op_peak_mem for w, op_peak_mem in zip(weights, peak_mem))

			#quantized option added
			peak_mem_list = [average_peak_mem / 8, average_peak_mem / 4, average_peak_mem / 2, average_peak_mem]
			#peak_mem_list = [peak_mem / 8, peak_mem / 4, peak_mem / 2, peak_mem]
			block_average_peak_mem = sum(g_w * active_mem for g_w, active_mem in zip(g_weights, peak_mem_list))
			diff_peak_mem = abs(block_average_peak_mem - self.target_mem)
			
			idx = torch.argmax(g_weights).item()

			if idx == 0:
				out_ = sum( w * self.activation_quantizer_4b(op(x)) for w, op in zip(weights, self.m_ops))
			elif idx == 1:
				out_ = sum( w * self.activation_quantizer_8b(op(x)) for w, op in zip(weights, self.m_ops))
			elif idx == 2:
				out_ = sum( w * self.activation_quantizer_16b(op(x)) for w, op in zip(weights, self.m_ops))
			elif idx == 3:
				out_ = sum( w * op(x) for w, op in zip(weights, self.m_ops))
			else:
				raise NotImplementedError

			return out_, diff_peak_mem
	
	def get_lookup_memory(self, size):
		peak_memory = []
		for idx, op in enumerate(self.m_ops):
			if isinstance(op, IdentityLayer):
				peak_memory.append(0)
			else:
				key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												op.name,
												size,
												op.in_channels,
												op.se_channels,
												op.out_channels,
												op.kernel_size,
												op.stride,
												op.act_func)
				mid_channels = op.mid_channels
				peak_memory.append(self.memory_lookup[key][mid_channels])

		return peak_memory		

	def _initialize_log_alphas(self):
		alphas = torch.zeros((self.num_ops,))
		log_alphas = F.log_softmax(alphas, dim=-1)
		self.register_parameter('log_alphas', nn.Parameter(log_alphas))

	def _initialize_gammas(self):
		gammas = torch.zeros(len(QUANTIZE_OPS))
		self.register_parameter('gammas', nn.Parameter(gammas))
	
	def reset_switches(self):
		self.switches = [True] * self.num_ops

	def set_temperature(self, T):
		self.T = T


class MixedStage(nn.Module):
	def __init__(self, ics, ocs, ss, affs, acts, mc_num_ddict, stage_type, stage_num, peakmemory_list, target_mem, memory_lookup):
		super(MixedStage, self).__init__()
		self.mc_num_ddict = mc_num_ddict
		self.stage_type = stage_type # 0 for stage5 || 1 for stage1 || 2 for stage2 || 3 for stage3/4
		self.stage_num = stage_num
		self.start_res = 0 if ((ics[0] == ocs[0]) and (ss[0] == 1)) else 1
		self.num_res = len(ics) - self.start_res + 1
		self.peakmemory_list = peakmemory_list

		# stage5
		if stage_type == 0:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
		# stage1
		elif stage_type == 1:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
		# stage2
		elif stage_type == 2:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES), mc_num_ddict['block3'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
		# stage3, stage4
		elif stage_type == 3:
			self.block1 = MixedOP(ics[0], ocs[0], ss[0], affs[0], acts[0], len(PRIMITIVES), mc_num_ddict['block1'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block2 = MixedOP(ics[1], ocs[1], ss[1], affs[1], acts[1], len(PRIMITIVES), mc_num_ddict['block2'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block3 = MixedOP(ics[2], ocs[2], ss[2], affs[2], acts[2], len(PRIMITIVES), mc_num_ddict['block3'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
			self.block4 = MixedOP(ics[3], ocs[3], ss[3], affs[3], acts[3], len(PRIMITIVES), mc_num_ddict['block4'], peakmemory_list = self.peakmemory_list, stage_num=stage_num, target_mem = target_mem, memory_lookup = memory_lookup)
		else:
			raise ValueError('invalid stage_type...')

		self._initialize_betas()

	def forward(self, x, sampling, mode):
		res_list = [x,]
		activation_list = [0.,]

		# stage5
		if self.stage_type == 0:
			# logging.info("stage5")
			# logging.info("block 1 ")
			out1, peak_mem = self.block1(x, sampling, mode)
			res_list.append(out1)
			activation_list.append(peak_mem)
			
		# stage 1
		elif self.stage_type == 1:
			# logging.info("stage1")
			# logging.info("block 1 ")
			out1, peak_mem = self.block1(x, sampling, mode)
			res_list.append(out1)
			activation_list.append(peak_mem)

			# logging.info("block 2")
			out2, peak_mem= self.block2(out1, sampling, mode)
			res_list.append(out2)
			activation_list.append(peak_mem)

		# stage2
		elif self.stage_type == 2:
			# logging.info("stage2")
			# logging.info("block 1")
			out1, peak_mem = self.block1(x, sampling, mode)
			res_list.append(out1)
			activation_list.append(peak_mem)

			# logging.info("block 2")
			out2, peak_mem= self.block2(out1, sampling, mode)
			res_list.append(out2)
			activation_list.append(peak_mem)

			# logging.info("block 3")
			out3, peak_mem = self.block3(out2, sampling, mode)
			res_list.append(out3)
			activation_list.append(peak_mem)

		# stage3, stage4
		elif self.stage_type == 3:
			#logging.info("stage3 / 4")
			#logging.info("block 1 ")
			out1, peak_mem = self.block1(x, sampling, mode)
			res_list.append(out1)
			activation_list.append(peak_mem)

			#logging.info("block 2 ")
			out2, peak_mem= self.block2(out1, sampling, mode)
			res_list.append(out2)
			activation_list.append(peak_mem)

			#logging.info("block 3 ")
			out3, peak_mem = self.block3(out2, sampling, mode)
			res_list.append(out3)
			activation_list.append(peak_mem)

			#logging.info("block 4 ")
			out4, peak_mem= self.block4(out3, sampling, mode)
			res_list.append(out4)
			activation_list.append(peak_mem)

		else:
			raise ValueError

		weights = F.softmax(self.betas, dim=-1)
		out = sum(w*res for w, res in zip(weights, res_list[self.start_res:]))
		 
		#out = sum(g_w * res for g_w, res in zip(g_weights, out_list))

		if sampling == False:
			peak_memory = sum(diff_between_peakmem_and_target for diff_between_peakmem_and_target in activation_list)			
			#print("stage_type :", self.stage_type, ", gamma_rate(n.q , q):", g_weights.tolist() , "weighted peak_mem:", peak_memory.item())
			return out, peak_memory
		else:
			return out, 0

	def _initialize_betas(self):
		betas = torch.zeros((self.num_res))
		self.register_parameter('betas', nn.Parameter(betas))


class Network(nn.Module):
	def __init__(self, num_classes, mc_num_dddict, target_mem, memory_lookup):
		super(Network, self).__init__()
		self.mc_num_dddict = mc_num_dddict
		self.peakmemory_list = [0.0, 0.0, 0.0, 0.0, 0.0]

		self.first_stem  = ConvLayer(3, 32, kernel_size=3, stride=2, affine=False, act_func='relu')
		self.second_stem = MBInvertedResBlock(32, 32, 8, 16, kernel_size=3, stride=1, affine=False, act_func='relu')
		self.stage1 = MixedStage(
							ics  = [16,24],
							ocs  = [24,24],
							ss   = [2,1],
							affs = [False, False],
							acts = ['relu', 'relu'],
							mc_num_ddict = mc_num_dddict['stage1'],
							stage_type = 1,
							stage_num = 1,
							peakmemory_list = self.peakmemory_list,
							target_mem = target_mem,
							memory_lookup = memory_lookup)
		self.stage2 = MixedStage(
							ics  = [24,40,40],
							ocs  = [40,40,40],
							ss   = [2,1,1],
							affs = [False, False, False],
							acts = ['swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage2'],
							stage_type = 2,
							stage_num = 2,
							peakmemory_list = self.peakmemory_list,
							target_mem = target_mem,
							memory_lookup = memory_lookup)
		self.stage3 = MixedStage(
							ics  = [40,80,80,80],
							ocs  = [80,80,80,80],
							ss   = [2,1,1,1],
							affs = [False, False, False, False],
							acts = ['swish', 'swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage3'],
							stage_type = 3,
							stage_num = 3,
							peakmemory_list = self.peakmemory_list,
							target_mem = target_mem,
							memory_lookup = memory_lookup)
		self.stage4 = MixedStage(
							ics  = [80,112,112,112],
							ocs  = [112,112,112,112],
							ss   = [1,1,1,1],
							affs = [False, False, False, False],
							acts = ['swish', 'swish', 'swish', 'swish'],
							mc_num_ddict = mc_num_dddict['stage4'],
							stage_type = 3,
							stage_num = 4,
							peakmemory_list = self.peakmemory_list,
							target_mem = target_mem,
							memory_lookup = memory_lookup)
		self.stage5 = MixedStage(
							ics  = [112,],
							ocs  = [320,],
							ss   = [1,],
							affs = [False,],
							acts = ['swish',],
							mc_num_ddict = mc_num_dddict['stage5'],
							stage_type = 0,
							stage_num = 5,
							peakmemory_list = self.peakmemory_list,
							target_mem = target_mem,
							memory_lookup = memory_lookup)
		self.feature_mix_layer = ConvLayer(320, 1280, kernel_size=1, stride=1, affine=False, act_func='swish')
		self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = LinearLayer(1280, num_classes)
		self._initialization()

	def forward(self, x, sampling, mode='max'):
		
		x = self.first_stem(x)
		x = self.second_stem(x)

		out_memory = 0.

		x, peak_mem = self.stage1(x, sampling, mode)
		out_memory += peak_mem

		x, peak_mem= self.stage2(x, sampling, mode)
		out_memory += peak_mem

		x, peak_mem = self.stage3(x, sampling, mode)
		out_memory += peak_mem

		x, peak_mem = self.stage4(x, sampling, mode)
		out_memory += peak_mem

		x, peak_mem = self.stage5(x, sampling, mode)
		out_memory += peak_mem

		x = self.feature_mix_layer(x)
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		
		return x, out_memory

	def set_temperature(self, T):
		for m in self.modules():
			if isinstance(m, MixedOP):
				m.set_temperature(T)

	def weight_parameters(self):
		_weight_parameters = []

		for k, v in self.named_parameters():
			if not (k.endswith('log_alphas') or k.endswith('betas') or k.endswith('gammas')):
				_weight_parameters.append(v)
		
		return _weight_parameters

	def arch_parameters(self):
		_arch_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('log_alphas') or k.endswith('betas'):
				_arch_parameters.append(v)

		return _arch_parameters

	def quantized_parameters(self):
		_quant_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('gammas'):
				_quant_parameters.append(v)

		return _quant_parameters

	def log_alphas_parameters(self):
		_log_alphas_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('log_alphas'):
				_log_alphas_parameters.append(v)

		return _log_alphas_parameters

	def betas_parameters(self):
		_betas_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('betas'):
				_betas_parameters.append(v)

		return _betas_parameters

	#for quantization 
	def gammas_parameters(self):
		_gammas_parameters = []

		for k, v in self.named_parameters():
			if k.endswith('gammas'):
				_gammas_parameters.append(v)
		
		return _gammas_parameters

	def reset_switches(self):
		for m in self.modules():
			if isinstance(m, MixedOP):
				m.reset_switches()

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

