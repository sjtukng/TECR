"""Tiny-Evidence-Guided Cross-Scale Refinement from TECR_JARS_V2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN


@NECKS.register_module(name='TECRV2FPN')
class TECRFPN(FPN):
    """The TECR refinement described in TECR_JARS_V2, on a standard FPN.

    Level indices 0..4 correspond to P2..P6. Evidence and gate predictors are
    shared across pyramid levels/edges. Every cross-scale tensor is resized to
    the receiver's explicit spatial size with bilinear interpolation.
    """

    def __init__(self, *args, enhanced_levels=(0, 1), lambda_init=0.1,
                 normalize_eps=1e-6, capture_debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_outs != 5:
            raise ValueError('TECR requires the five FPN levels P2-P6')
        if any(i < 0 or i >= self.num_outs for i in enhanced_levels):
            raise ValueError('enhanced_levels must index P2-P6 as 0..4')

        channels = self.out_channels
        self.enhanced_levels = tuple(enhanced_levels)
        self.normalize_eps = float(normalize_eps)
        self.capture_debug = bool(capture_debug)
        self.last_debug = {}

        self.evidence_head = nn.Conv2d(2 * channels, 1, 1)
        self.local_convs = nn.ModuleDict({
            str(i): nn.Conv2d(channels, channels, 3, padding=1,
                              groups=channels)
            for i in self.enhanced_levels
        })
        self.lambdas = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(float(lambda_init)))
            for i in self.enhanced_levels
        })

        self.projections = nn.ModuleDict()
        self.theta = nn.ParameterDict()
        for target in range(self.num_outs):
            for source in self._neighbors(target):
                key = self._edge(source, target)
                self.projections[key] = nn.Conv2d(channels, channels, 1)
                self.theta[key] = nn.Parameter(torch.zeros(()))

        # [receiver evidence, aligned source evidence, cosine consistency]
        self.gate_head = nn.Conv2d(3, 1, 3, padding=1)
        self._init_tecr_weights()

    def init_weights(self):
        super().init_weights()
        self._init_tecr_weights()

    def _init_tecr_weights(self):
        nn.init.xavier_uniform_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        for conv in self.local_convs.values():
            nn.init.kaiming_normal_(conv.weight, mode='fan_in',
                                    nonlinearity='linear')
            nn.init.zeros_(conv.bias)
        for conv in self.projections.values():
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.normal_(self.gate_head.weight, std=0.01)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _edge(source, target):
        return '{}_to_{}'.format(source, target)

    def _neighbors(self, target):
        return [source for source in (target - 1, target + 1)
                if 0 <= source < self.num_outs]

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode='bilinear',
                             align_corners=False)

    def _evidence(self, feature):
        residual = feature - F.avg_pool2d(
            feature, kernel_size=3, stride=1, padding=1)
        return torch.sigmoid(self.evidence_head(
            torch.cat((feature, residual.abs()), dim=1)))

    def _consistency(self, receiver, message):
        # Equation (8) uses norm + epsilon, rather than the clamped norm used
        # by F.normalize. The distinction matters for low-amplitude features.
        receiver = receiver / (
            receiver.norm(p=2, dim=1, keepdim=True) + self.normalize_eps)
        message = message / (
            message.norm(p=2, dim=1, keepdim=True) + self.normalize_eps)
        return (1.0 + (receiver * message).sum(dim=1, keepdim=True)) * 0.5

    @auto_fp16()
    def forward(self, inputs):
        pyramid = super().forward(inputs)
        evidence = [self._evidence(feature) for feature in pyramid]

        enhanced = list(pyramid)
        for level in self.enhanced_levels:
            enhanced[level] = (
                pyramid[level] + self.lambdas[str(level)] * evidence[level]
                * self.local_convs[str(level)](pyramid[level]))

        outputs, gates, consistency, mixing = [], {}, {}, {}
        # All directed messages are formed from the completed enhanced pyramid.
        for target, receiver in enumerate(enhanced):
            neighbors = self._neighbors(target)
            logits = torch.stack([
                self.theta[self._edge(source, target)]
                for source in neighbors
            ])
            weights = torch.softmax(logits, dim=0)
            result = receiver
            for index, source in enumerate(neighbors):
                key = self._edge(source, target)
                size = receiver.shape[-2:]
                aligned_feature = self._resize(enhanced[source], size)
                message = self.projections[key](aligned_feature)
                aligned_evidence = self._resize(evidence[source], size)
                cue = self._consistency(receiver, message)
                gate = torch.sigmoid(self.gate_head(torch.cat(
                    (evidence[target], aligned_evidence, cue), dim=1)))
                result = result + weights[index] * gate * message
                gates[key], consistency[key], mixing[key] = (
                    gate, cue, weights[index])
            outputs.append(result)

        if self.capture_debug:
            self.last_debug = dict(
                evidence=[value.detach() for value in evidence],
                gates={key: value.detach() for key, value in gates.items()},
                consistency={key: value.detach()
                             for key, value in consistency.items()},
                mixing={key: value.detach() for key, value in mixing.items()})
        return tuple(outputs)
