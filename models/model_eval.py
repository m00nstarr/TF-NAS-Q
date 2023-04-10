import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

from .quantizers import *
from .range_trackers import *

PRIMITIVES = [
        'MBI_k3_e3',
        'MBI_k3_e6',
        'MBI_k5_e3',
        'MBI_k5_e6',
        # 'skip',
        ]

OPS = {
        'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
        'MBI_k3_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
        'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
        'MBI_k5_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
        # 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
        }


class Network(nn.Module):
    def __init__(self, num_classes, parsed_arch, mc_num_dddict, lat_lookup=None, dropout_rate=0.0, drop_connect_rate=0.0):
        super(Network, self).__init__()
        self.lat_lookup = lat_lookup
        self.mc_num_dddict = mc_num_dddict
        self.parsed_arch = parsed_arch
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.block_count = self._get_block_count()
        self.block_idx = 0

        self.first_stem  = ConvLayer(3, 32, kernel_size=3, stride=2, affine=True, act_func='relu')
        self.second_stem = MBInvertedResBlock(32, 32, 8, 16, kernel_size=3, stride=1, affine=True, act_func='relu')
        self.block_idx += 1
        self.second_stem.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
        self.stage1 = self._make_stage('stage1',
                ics  = [16,24],
                ocs  = [24,24],
                ss   = [2,1],
                affs = [True, True],
                acts = ['relu', 'relu'],)
        self.stage2 = self._make_stage('stage2',
                ics  = [24,40,40],
                ocs  = [40,40,40],
                ss   = [2,1,1],
                affs = [True, True, True],
                acts = ['swish', 'swish', 'swish'],)
        self.stage3 = self._make_stage('stage3',
                ics  = [40,80,80,80],
                ocs  = [80,80,80,80],
                ss   = [2,1,1,1],
                affs = [True, True, True, True],
                acts = ['swish', 'swish', 'swish', 'swish'],)
        self.stage4 = self._make_stage('stage4',
                ics  = [80,112,112,112],
                ocs  = [112,112,112,112],
                ss   = [1,1,1,1],
                affs = [True, True, True, True],
                acts = ['swish', 'swish', 'swish', 'swish'],)
        self.stage5 = self._make_stage('stage5',
                ics  = [112,],
                ocs  = [320,],
                ss   = [1,],
                affs = [True,],
                acts = ['swish',],)
        self.feature_mix_layer = ConvLayer(320, 1280, kernel_size=1, stride=1, affine=True, act_func='swish')
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(1280, num_classes)
        self._initialization()


    def _get_block_count(self):
        count = 1
        for stage in self.parsed_arch:
            count += len(self.parsed_arch[stage])

        return count

    def _make_stage(self, stage_name, ics, ocs, ss, affs, acts):
        stage = nn.ModuleList()
        for i, block_name in enumerate(self.parsed_arch[stage_name]):
            self.block_idx += 1
            op_idx = self.parsed_arch[stage_name][block_name][0]
            primitive = PRIMITIVES[op_idx]
            mc = self.mc_num_dddict[stage_name][block_name][op_idx]
            op = OPS[primitive](ics[i], mc, ocs[i], ss[i], affs[i], acts[i])
            op.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
            stage.append(op)

        return stage

    def forward(self, x):
        x = self.first_stem(x)
        x = self.second_stem(x, 2)

        for block_idx, block in enumerate(self.stage1):
            blk_key = 'block{}'.format(block_idx+1)
            q = self.parsed_arch['stage1'][blk_key][1]
            x = block(x, q)

        for block_idx, block in enumerate(self.stage2):
            blk_key = 'block{}'.format(block_idx+1)
            q = self.parsed_arch['stage2'][blk_key][1]
            x = block(x, q)

        for block_idx, block in enumerate(self.stage3):
            blk_key = 'block{}'.format(block_idx+1)
            q = self.parsed_arch['stage3'][blk_key][1]
            x = block(x, q)

        for block_idx, block in enumerate(self.stage4):
            blk_key = 'block{}'.format(block_idx+1)
            q = self.parsed_arch['stage4'][blk_key][1]
            x = block(x, q)

        for block_idx, block in enumerate(self.stage5):
            blk_key = 'block{}'.format(block_idx+1)
            q = self.parsed_arch['stage5'][blk_key][1]
            x = block(x, q)

        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        return x

    @property
    def config(self):
        return { 'first_stem':  self.first_stem.config,
                'second_stem': self.second_stem.config,
                'stage1': [block.config for block in self.stage1],
                'stage2': [block.config for block in self.stage2],
                'stage3': [block.config for block in self.stage3],
                'stage4': [block.config for block in self.stage4],
                'stage5': [block.config for block in self.stage5],
                'feature_mix_layer': self.feature_mix_layer.config,
                'classifier': self.classifier.config,
                }

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
