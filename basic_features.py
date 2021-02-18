import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_dir = './pretrained_models'



class Basic_features(nn.Module):

    def __init__(self):
        super(Basic_features, self).__init__()

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
        self.batch_norm = False
        p = 0.25
        self.drop_layer1 = nn.Dropout(p=p)
        self.drop_layer2 = nn.Dropout(p=p)
        self.features = self._make_layers()
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self):

        self.n_layers = 0

        layers = []
        in_channels = 1
        conv2d = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.kernel_sizes.append(1)
        self.strides.append(1)
        self.paddings.append(2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 32
        conv2d = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.kernel_sizes.append(5)
        self.strides.append(1)
        self.paddings.append(2)
        in_channels = 32
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2)]
        self.kernel_sizes.append(2)
        self.strides.append(2)
        self.paddings.append(0)
        layers += [self.drop_layer1]

        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.kernel_sizes.append(3)
        self.strides.append(1)
        self.paddings.append(1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.kernel_sizes.append(3)
        self.strides.append(1)
        self.paddings.append(1)
        in_channels = 64
        layers += [conv2d, nn.ReLU(inplace=True)]
        
        layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        self.kernel_sizes.append(2)
        self.strides.append(2)
        self.paddings.append(0)
        layers += [self.drop_layer2]

        return nn.Sequential(*layers)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    def __repr__(self):
        template = 'VGG{}, batch_norm={}'
        return template.format(self.num_layers() + 3,
                               self.batch_norm)



def basic_features(**kwargs):
    """Basic model (configuration "A")
    Based on: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = Basic_features()
    return model



if __name__ == '__main__':

    basic = basic_features()
    print(basic)

