import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader

import argparse
import re
import pandas as pd

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
import numpy as np
import pickle
#from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-seedL', nargs=1, type=int, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-seedH', nargs=1, type=int, default='30') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run
percentage = 0.7
for seed in range(args.seedL[0],args.seedH[0]):
    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = './saved_models/' + base_architecture +"_memory"+ '/' + str(seed) + '/'
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    from settings import train_dir, test_dir, train_push_dir, \
                         train_batch_size, test_batch_size, train_push_batch_size

    #normalize = transforms.Normalize(mean=mean,
    #                                 std=std)

    # all datasets
    # train set
    #train_dataset = datasets.ImageFolder(
    #    train_dir,
    #    transforms.Compose([
    #        transforms.Grayscale(), 
    #        transforms.Resize(size=(img_size, img_size)),
    #        transforms.ToTensor(),
    #       normalize,
    #    ]))

    train = pd.read_csv("../data/MNIST/mnist_train.csv")
    test = pd.read_csv("../data/MNIST/mnist_test.csv")

    y_train = train["label"]
    y_test = test["label"]
    # Drop 'label' column
    x_train = train.drop(labels = ["label"],axis = 1) 
    x_train = x_train.values.reshape(-1,1,28,28)
    x_test = test.drop(labels = ["label"],axis = 1) 
    x_test = x_test.values.reshape(-1,1,28,28)

    '''
    def new_training_set(seed,percentage):
        np.random.seed(seed)
        indices = np.random.choice(len(x_train), size=int(percentage*len(x_train)), replace=False)
        train_tensor_x = torch.FloatTensor(x_train[indices])
        train_tensor_y = torch.LongTensor(np.array(y_train[indices]))
        train_tensor_x.cuda()
        train_tensor_y.cuda()

        train_dataset = TensorDataset(train_tensor_x,train_tensor_y) 
        return train_dataset
'''

    train_tensor_x = torch.FloatTensor(x_train) # transform to torch tensor
    train_tensor_y = torch.LongTensor(y_train)
    train_tensor_x.cuda()
    train_tensor_y.cuda()


    full_train_dataset = TensorDataset(train_tensor_x,train_tensor_y) 

    test_tensor_x = torch.FloatTensor(x_test) # transform to torch tensor
    test_tensor_y = torch.LongTensor(y_test)
    test_tensor_x.cuda()
    test_tensor_y.cuda()

    #train_dataset = new_training_set(seed,percentage) # 
    test_dataset = TensorDataset(test_tensor_x,test_tensor_y) 

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=train_batch_size, shuffle=True,
    #    num_workers=4, pin_memory=False)
    # push set
    #train_push_dataset = datasets.ImageFolder(
    #    train_push_dir,
    #    transforms.Compose([
    #        transforms.Grayscale(), 
    #        transforms.Resize(size=(img_size, img_size)),
    #        transforms.ToTensor(),
    #    ]))
    #train_push_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=train_push_batch_size, shuffle=False,
    #    num_workers=4, pin_memory=False)
    # test set
    #test_dataset = datasets.ImageFolder(
    #    test_dir,
    #    transforms.Compose([
    #        transforms.Grayscale(), 
    #        transforms.Resize(size=(img_size, img_size)),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    #log('training set size: {0}'.format(len(train_loader.dataset)))
    #log('push set size: {0}'.format(len(train_push_loader.dataset)))
    #log('test set size: {0}'.format(len(test_loader.dataset)))
    #log('batch size: {0}'.format(train_batch_size))

    # construct the model
    #ppnet = model.construct_PPNet(base_architecture=base_architecture,
    #                              pretrained=True, img_size=img_size,
    #                              prototype_shape=prototype_shape,
    #                              num_classes=num_classes,
    #                              prototype_activation_function=prototype_activation_function,
    #                              add_on_layers_type=add_on_layers_type)
    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            load_model_path = os.path.join(model_dir, file)
    
    ppnet = torch.load(load_model_path)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    '''
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 0, 'alpha': 0.9, 'eps':1e-08}, # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 0, 'alpha': 0.9, 'eps':1e-08},
     {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.RMSprop(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from settings import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

    # train the model
    log('start training')
    import copy
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
        #                            target_accu=0.70, log=log)

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=None, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
            #                            target_accu=0.70, log=log)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                     #                           target_accu=0.70, log=log)
    # Get the predictions 
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.70, log=log)
    '''
    prediction_dataloader = DataLoader(full_train_dataset,batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False) # create your dataloader
    prediction = tnt.prediction(model=ppnet_multi, dataloader=prediction_dataloader,
                        class_specific=class_specific, log=log, evaluation_pred="test.pkl")
    pickle.dump(np.asarray(prediction), open(model_dir+"new_train_predictions.pkl","wb"))

    test_prediction = tnt.prediction(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log, evaluation_pred="test.pkl")
    pickle.dump(np.asarray(test_prediction), open(model_dir+"new_test_predictions.pkl","wb"))

logclose()

