# -*- coding: utf-8 -*-
from models.graphunet import GraphUNet, GraphNet
from models.resnet import resnet10, resnet18, resnet50, resnet101
from models.hopenet import HopeNet
from models.contextnet import ContextModule, ContextNet

def select_model(model_def):
    if model_def.lower() == 'hopenet':
        model = HopeNet()
        print('HopeNet is created')
    elif model_def.lower() == 'resnet10':
        model = resnet10(pretrained=False, num_classes=29*2)
        print('ResNet10 is created')
    elif model_def.lower() == 'resnet18':
        model = resnet18(pretrained=False, num_classes=29*2)
        print('ResNet18 is created')
    elif model_def.lower() == 'resnet50':
        model = resnet50(pretrained=False, num_classes=29*2)
        print('ResNet50 is created')
    elif model_def.lower() == 'resnet101':
        model = resnet101(pretrained=False, num_classes=29*2)
        print('ResNet101 is created')
    elif model_def.lower() == 'graphunet':
        model = GraphUNet(in_features=2, out_features=3)
        print('GraphUNet is created')
    elif model_def.lower() == 'graphnet':
        model = GraphNet(in_features=2, out_features=3)
        print('GraphNet is created')
    elif model_def.lower() == 'contextnet':
        model = ContextNet()
        print('ContextNet is created')
    elif model_def.lower() == 'contextmodule':
        model = ContextModule(in_features=2, out_features=3)
        print('ContextModule is created')
    else:
        raise NameError('Undefined model')
    return model
