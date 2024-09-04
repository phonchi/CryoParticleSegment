import torch
import torch.nn as nn
from collections import OrderedDict
from convcrf import GaussCRF, ConvCRF


def create_model(backbone, addout=True):
    if addout:
        model = Model_Out(backbone)
    else:
        model = backbone
    #model.classifier.add_module(name='output', module=nn.Softmax(dim=1))
    return model


def create_crf_model(backbone, config, shape, num_classes, use_gpu=False, freeze_backbone=False):
    if freeze_backbone:
        for params in backbone.parameters():
          params.requires_grad = False
    model = ModelWithGausscrf(backbone, config=config, shape=shape, num_classes=num_classes, use_gpu=use_gpu)
    return model


class Model_Out(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        """
        x is a batch of input images
        """
        logits = self.backbone(x)
        return OrderedDict([
        ('out', logits)
      ])

class ModelWithGausscrf(nn.Module):
    def __init__(self, backbone, config, shape, num_classes, use_gpu=False):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.num_classes = num_classes
        self.shape = shape
        self.use_gpu = use_gpu
        self.gausscrf = GaussCRF(conf=self.config, shape=self.shape,
                                 nclasses=self.num_classes, use_gpu= self.use_gpu)

    def forward(self, x):
        unary = self.backbone(x)['out']
        return OrderedDict([
          ('backbone', unary),
          ('out', self.gausscrf(unary, x))
        ])


try:
    import CRF
    class ModelWithFWCRF(nn.Module):
        """Combined Model class for UNET with configurable Frank-Wolfe CRF."""
        def __init__(self, backbone, crf, use_unary_only=False):
            super().__init__()
            self.backbone = backbone
            self.crf = crf
            self.use_unary_only = use_unary_only

        def forward(self, x):
            """Forward pass for input batch of images."""
            unary = self.backbone(x)
            if self.use_unary_only:
                return {'backbone': unary, 'out': self.crf(unary, unary)}
            else:
                return {'backbone': unary, 'out': self.crf(x, unary)}

    def create_fwcrf_model(backbone, crf, use_unary_only=False):
        """Factory function to create a UNET model with Frank-Wolfe CRF."""
        return ModelWithFWCRF(backbone, crf, use_unary_only)

    def setup_crf(solver, num_classes):
        """Setup CRF based on the solver type."""
        if solver not in ['fw', 'mf']:
            raise NotImplementedError("Solver not supported")
        
        crf = CRF.DenseGaussianCRF(
            classes=num_classes,
            alpha=160,
            beta=0.05,
            gamma=3.0,
            spatial_weight=1.0,
            bilateral_weight=1.0,
            compatibility=1.0,
            init='potts',
            solver=solver,
            iterations=5,
            params=None if solver == 'mf' else CRF.FrankWolfeParams(
                scheme='fixed', stepsize=1.0, regularizer='l2', lambda_=1.0,
                lambda_learnable=False, x0_weight=0.5, x0_weight_learnable=False)
        )
        return crf
    
  # class ModelWithFWCRF(nn.Module):
      # def __init__(self, backbone, crf):
          # super().__init__()
          # self.backbone = backbone
          # self.crf = crf

      # def forward(self, x):
          # """
          # x is a batch of input images
          # """
          # unary = self.backbone(x)['out']
          # logits = self.crf(x, unary)
          # return OrderedDict([
          # ('backbone', unary),
          # ('out', logits)
        # ])
        
  # class ModelWithFWCRF_UNET(nn.Module):
      # def __init__(self, backbone, crf):
          # super().__init__()
          # self.backbone = backbone
          # self.crf = crf

      # def forward(self, x):
          # """
          # x is a batch of input images
          # """
          # unary = self.backbone(x)
          # logits = self.crf(x, unary)
          # return OrderedDict([
          # ('backbone', unary),
          # ('out', logits)
        # ])

  # def create_fwcrf_model_unet(backbone, params, num_classes, alpha=160, beta=0.05, gamma=3.0, iterations=5, freeze_backbone=False):
    # if freeze_backbone:
      # for param in backbone.parameters():
        # param.requires_grad = False
    # crf = CRF.DenseGaussianCRF(
            # classes=num_classes,
            # alpha=alpha,
            # beta=beta,
            # gamma=gamma,
            # spatial_weight=1.0,
            # bilateral_weight=1.0,
            # compatibility=1.0,
            # init='potts',
            # solver='mf',
            # iterations=iterations,
            # x0_weight = 0,
            # params=params)
    # model = ModelWithFWCRF_UNET(backbone, crf)
    # return model

  # def create_fwcrf_model(backbone, params, num_classes, alpha=160, beta=0.05, gamma=3.0, iterations=5, freeze_backbone=False):
    # if freeze_backbone:
      # for param in backbone.parameters():
        # param.requires_grad = False
    # crf = CRF.DenseGaussianCRF(
            # classes=num_classes,
            # alpha=alpha,
            # beta=beta,
            # gamma=gamma,
            # spatial_weight=1.0,
            # bilateral_weight=1.0,
            # compatibility=1.0,
            # init='potts',
            # solver='mf',
            # iterations=iterations,
            # x0_weight = 0,
            # params=params)
    # model = ModelWithFWCRF(backbone, crf)
    # return model
except:
    pass