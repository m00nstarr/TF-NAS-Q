import torch
from collections import OrderedDict

mc_mask_dddict = OrderedDict([
		('stage1', OrderedDict([
				('block1', OrderedDict([
						(0, torch.cat((torch.ones(16*3),torch.zeros(16*1)))),
						(1, torch.cat((torch.ones(16*6),torch.zeros(16*2)))),
						(2, torch.cat((torch.ones(16*3),torch.zeros(16*1)))),
						(3, torch.cat((torch.ones(16*6),torch.zeros(16*2)))),
						# (4, torch.cat((torch.ones(16*3),torch.zeros(16*1)))),
						# (5, torch.cat((torch.ones(16*6),torch.zeros(16*2)))),
						# (6, torch.cat((torch.ones(16*3),torch.zeros(16*1)))),
						# (7, torch.cat((torch.ones(16*6),torch.zeros(16*2)))),
					])),
				('block2', OrderedDict([
						(0, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						(1, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						(2, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						(3, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						# (4, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						# (5, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						# (6, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						# (7, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
					])),
			])),
		('stage2', OrderedDict([
				('block1', OrderedDict([
						(0, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						(1, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						(2, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						(3, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						# (4, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						# (5, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
						# (6, torch.cat((torch.ones(24*3),torch.zeros(24*1)))),
						# (7, torch.cat((torch.ones(24*6),torch.zeros(24*2)))),
					])),
				('block2', OrderedDict([
						(0, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(1, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						(2, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(3, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (4, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (5, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (6, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (7, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
					])),
				('block3', OrderedDict([
						(0, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(1, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						(2, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(3, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (4, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (5, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (6, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (7, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
					])),
			])),
		('stage3', OrderedDict([
				('block1', OrderedDict([
						(0, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(1, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						(2, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						(3, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (4, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (5, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
						# (6, torch.cat((torch.ones(40*3),torch.zeros(40*1)))),
						# (7, torch.cat((torch.ones(40*6),torch.zeros(40*2)))),
					])),
				('block2', OrderedDict([
						(0, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(1, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						(2, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(3, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (4, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (5, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (6, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (7, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
					])),
				('block3', OrderedDict([
						(0, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(1, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						(2, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(3, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (4, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (5, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (6, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (7, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
					])),
				('block4', OrderedDict([
						(0, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(1, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						(2, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(3, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (4, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (5, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (6, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (7, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
					])),
			])),
		('stage4', OrderedDict([
				('block1', OrderedDict([
						(0, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(1, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						(2, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						(3, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (4, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (5, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
						# (6, torch.cat((torch.ones(80*3),torch.zeros(80*1)))),
						# (7, torch.cat((torch.ones(80*6),torch.zeros(80*2)))),
					])),
				('block2', OrderedDict([
						(0, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(1, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						(2, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(3, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (4, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (5, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (6, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (7, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
					])),
				('block3', OrderedDict([
						(0, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(1, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						(2, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(3, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (4, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (5, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (6, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (7, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
					])),
				('block4', OrderedDict([
						(0, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(1, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						(2, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(3, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (4, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (5, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (6, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						# (7, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
					])),
			])),
		('stage5', OrderedDict([
				('block1', OrderedDict([
						(0, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(1, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						(2, torch.cat((torch.ones(112*3),torch.zeros(112*1)))),
						(3, torch.cat((torch.ones(112*6),torch.zeros(112*2)))),
						# (4, torch.cat((torch.ones(192*3),torch.zeros(192*1)))),
						# (5, torch.cat((torch.ones(192*6),torch.zeros(192*2)))),
						# (6, torch.cat((torch.ones(192*3),torch.zeros(192*1)))),
						# (7, torch.cat((torch.ones(192*6),torch.zeros(192*2)))),
					])),
			])),
	])


lat_lookup_key_dddict = OrderedDict([
		('stage1', OrderedDict([
				('block1', OrderedDict([
						(0, 'MBInvertedResBlock_112_16_0_24_k3_s2_relu'),
						(1, 'MBInvertedResBlock_112_16_0_24_k3_s2_relu'),
						(2, 'MBInvertedResBlock_112_16_0_24_k5_s2_relu'),
						(3, 'MBInvertedResBlock_112_16_0_24_k5_s2_relu'),
						# (4, 'MBInvertedResBlock_112_16_16_24_k3_s2_relu'),
						# (5, 'MBInvertedResBlock_112_16_32_24_k3_s2_relu'),
						# (6, 'MBInvertedResBlock_112_16_16_24_k5_s2_relu'),
						# (7, 'MBInvertedResBlock_112_16_32_24_k5_s2_relu'),
					])),
				('block2', OrderedDict([
						(0, 'MBInvertedResBlock_56_24_0_24_k3_s1_relu'),
						(1, 'MBInvertedResBlock_56_24_0_24_k3_s1_relu'),
						(2, 'MBInvertedResBlock_56_24_0_24_k5_s1_relu'),
						(3, 'MBInvertedResBlock_56_24_0_24_k5_s1_relu'),
						# (4, 'MBInvertedResBlock_56_24_24_24_k3_s1_relu'),
						# (5, 'MBInvertedResBlock_56_24_48_24_k3_s1_relu'),
						# (6, 'MBInvertedResBlock_56_24_24_24_k5_s1_relu'),
						# (7, 'MBInvertedResBlock_56_24_48_24_k5_s1_relu'),
					])),
			])),
		('stage2', OrderedDict([
				('block1', OrderedDict([
						(0, 'MBInvertedResBlock_56_24_0_40_k3_s2_swish'),
						(1, 'MBInvertedResBlock_56_24_0_40_k3_s2_swish'),
						(2, 'MBInvertedResBlock_56_24_0_40_k5_s2_swish'),
						(3, 'MBInvertedResBlock_56_24_0_40_k5_s2_swish'),
						# (4, 'MBInvertedResBlock_56_24_24_40_k3_s2_swish'),
						# (5, 'MBInvertedResBlock_56_24_48_40_k3_s2_swish'),
						# (6, 'MBInvertedResBlock_56_24_24_40_k5_s2_swish'),
						# (7, 'MBInvertedResBlock_56_24_48_40_k5_s2_swish'),
					])),
				('block2', OrderedDict([
						(0, 'MBInvertedResBlock_28_40_0_40_k3_s1_swish'),
						(1, 'MBInvertedResBlock_28_40_0_40_k3_s1_swish'),
						(2, 'MBInvertedResBlock_28_40_0_40_k5_s1_swish'),
						(3, 'MBInvertedResBlock_28_40_0_40_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_28_40_40_40_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_28_40_80_40_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_28_40_40_40_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_28_40_80_40_k5_s1_swish'),
					])),
				('block3', OrderedDict([
						(0, 'MBInvertedResBlock_28_40_0_40_k3_s1_swish'),
						(1, 'MBInvertedResBlock_28_40_0_40_k3_s1_swish'),
						(2, 'MBInvertedResBlock_28_40_0_40_k5_s1_swish'),
						(3, 'MBInvertedResBlock_28_40_0_40_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_28_40_40_40_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_28_40_80_40_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_28_40_40_40_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_28_40_80_40_k5_s1_swish'),
					])),
			])),
		('stage3', OrderedDict([
				('block1', OrderedDict([
						(0, 'MBInvertedResBlock_28_40_0_80_k3_s2_swish'),
						(1, 'MBInvertedResBlock_28_40_0_80_k3_s2_swish'),
						(2, 'MBInvertedResBlock_28_40_0_80_k5_s2_swish'),
						(3, 'MBInvertedResBlock_28_40_0_80_k5_s2_swish'),
						# (4, 'MBInvertedResBlock_28_40_40_80_k3_s2_swish'),
						# (5, 'MBInvertedResBlock_28_40_80_80_k3_s2_swish'),
						# (6, 'MBInvertedResBlock_28_40_40_80_k5_s2_swish'),
						# (7, 'MBInvertedResBlock_28_40_80_80_k5_s2_swish'),
					])),
				('block2', OrderedDict([
						(0, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_80_80_80_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_80_160_80_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_80_80_80_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_80_160_80_k5_s1_swish'),
					])),
				('block3', OrderedDict([
						(0, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_80_80_80_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_80_160_80_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_80_80_80_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_80_160_80_k5_s1_swish'),
					])),
				('block4', OrderedDict([
						(0, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_80_0_80_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_80_0_80_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_80_80_80_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_80_160_80_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_80_80_80_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_80_160_80_k5_s1_swish'),
					])),
			])),
		('stage4', OrderedDict([
				('block1', OrderedDict([
						(0, 'MBInvertedResBlock_14_80_0_112_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_80_0_112_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_80_0_112_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_80_0_112_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_80_80_112_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_80_160_112_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_80_80_112_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_80_160_112_k5_s1_swish'),
					])),
				('block2', OrderedDict([
						(0, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_112_112_112_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_112_224_112_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_112_112_112_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_112_224_112_k5_s1_swish'),
					])),
				('block3', OrderedDict([
						(0, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_112_112_112_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_112_224_112_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_112_112_112_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_112_224_112_k5_s1_swish'),
					])),
				('block4', OrderedDict([
						(0, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_112_0_112_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_112_0_112_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_14_112_112_112_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_14_112_224_112_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_14_112_112_112_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_14_112_224_112_k5_s1_swish'),
					])),
			])),
		('stage5', OrderedDict([
				('block1', OrderedDict([
						(0, 'MBInvertedResBlock_14_112_0_320_k3_s1_swish'),
						(1, 'MBInvertedResBlock_14_112_0_320_k3_s1_swish'),
						(2, 'MBInvertedResBlock_14_112_0_320_k5_s1_swish'),
						(3, 'MBInvertedResBlock_14_112_0_320_k5_s1_swish'),
						# (4, 'MBInvertedResBlock_7_192_192_320_k3_s1_swish'),
						# (5, 'MBInvertedResBlock_7_192_384_320_k3_s1_swish'),
						# (6, 'MBInvertedResBlock_7_192_192_320_k5_s1_swish'),
						# (7, 'MBInvertedResBlock_7_192_384_320_k5_s1_swish'),
					])),
			])),
	])
