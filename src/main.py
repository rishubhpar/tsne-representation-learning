import torch
import numpy as np
import pandas as pd
import sklearn
import os
import torchvision
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as trans
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tsne_dataset import tsne_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from models import feature_extractor, classifier_model, conv_feature_extractor, conv_classifier_model
import seaborn as sns
import matplotlib.pyplot as plt 

def read_mnist(split):
    transforms = trans.Compose([trans.ToTensor(), trans.Lambda(lambda x: torch.flatten(x))])
                                     
    mnist = MNIST(root='../data/mnist/', train=split, transform = transforms, download=False) 

    """    
    for i in range(0, 10):
        x,y = mnist[i]
        print(x.shape, y)
    """
    # mnist is a list of x,y pairs, where x is flatten 784 pixels and y is the class label index

    return mnist

def read_cifar(split):
    transforms = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), trans.Lambda(lambda x: torch.flatten(x))])
    # transforms = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar = CIFAR10(root='../data/cifar10', train=split, transform = transforms, download=True)
    """
    for i in range(0, 10):
        x,y = cifar[i]
        print(x.shape, y)
    """
    return cifar


def read_svhn(split):
    transforms = trans.Compose([trans.ToTensor(), trans.Lambda(lambda x: torch.flatten(x))])
    # transforms = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar = SVHN(root='../data/svhn', split=split, transform = transforms, download=True)
    """
    for i in range(0, 10):
        x,y = cifar[i]
        print(x.shape, y)
    """
    return cifar

# Function to read appropriate dataset 
def read_data(dataset, split):
    if (dataset == 'cifar10'):
        data = read_cifar(split)
    elif (dataset == 'mnist'):
        data = read_mnist(split)
    return data


def save_ckpt(model, save_path, epoch):
    # Saving the checkpoint with currect epoch number
    ckpt_path = save_path + '_ep' + str(epoch) + '.pth'
    model_save_bundle = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(), 
            }

    torch.save(model_save_bundle, ckpt_path)

# Plotting the tsne by first computing the PCA of tsne dimensions and plotting them
def plot_mappings(tsne_data, save_name):
    plt.figure(figsize=(16,10))
    
    pca = PCA(n_components=2) 
    pca_result = pca.fit_transform(tsne_data['x'])  
    y = tsne_data['y']
    pca_one = pca_result[:,0]
    pca_two = pca_result[:,1]

    print("Plotting {} datapoints in PCA".format(len(y)))
    data = pd.DataFrame()
    data['y'] = y
    data['pca_one'] = pca_one
    data['pca_two'] = pca_two

    sns.scatterplot(
        x='pca_one', y='pca_two',
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.3
    )
    plt.show()
    plt.savefig(save_name)
    plt.clf()

def compute_tsne(dataset, num_points, params):
    print("computing tsne for datset shape: ", num_points)

    # Creating a numpy matrix for x
    xs, ys = [], []
    for i in range(num_points):
        x,y = dataset[i]
        xs.append(x.numpy())
        ys.append(y)

    xs = np.array(xs)
    print("XS shape:", xs.shape, " xs type:", xs.dtype) 

    # tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=600, method='exact')
    tsne = TSNE(n_components=params['tsne_dimensions'], verbose=1, perplexity=40, n_iter=params['tsne_iters'], method='exact')
    tsne_transform_x = tsne.fit_transform(xs)

    tsne_transform = {'x': tsne_transform_x, 'y': ys}
    print("compute tsne transforms shape: ", tsne_transform['x'].shape, len(tsne_transform['y'])) 

    # Plotting pca for the tsne-embeddings to visualize
    plot_mappings(tsne_transform, 'original_svhn_tsne_pca_6000s.png') 
    
    # Saving tnse-transforms
    np.save(params['tsne_save_path'], tsne_transform['x']) 
    return tsne_transform_x

def query_tsne(dataset, params):
    tsne_path = params['tsne_save_path']
    if (os.path.exists(tsne_path)):
        tsne_transform = np.load(tsne_path)
    else:
        tsne_transform = compute_tsne(dataset, params['tsne_data_size'], params)

    return tsne_transform

def evaluate_tsne_predictions(model, dataset_train, dataset_test, tsne_gt, save_prefix, epoch_pt):
    train_pred_embds = []
    train_labels = []

    # Computing forward pass on the train set
    for i, batch in enumerate(dataset_train):
        x, y = batch
        x = x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])
        x = x.cuda()
        embds_ = model(x)[0] # Selecting the first and only element from the batch of examples
        # print("feature output shape: ", embds_.shape)
        train_pred_embds.append(embds_.detach().cpu().numpy())
        train_labels.append(y)

    # print("Debug : ", len(pred_embds), " shape : ", pred_embds[10].shape)
    train_pred_embds = np.array(train_pred_embds)


    test_pred_embds = []
    test_labels = []

    # Computing forward pass on the test set
    for i, batch in enumerate(dataset_test):
        x, y = batch
        x = x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])   
        x = x.cuda()
        embds_ = model(x)[0] # Selecting the first and only element from the batch of examples
        # print("feature output shape: ", embds_.shape)

        test_pred_embds.append(embds_.detach().cpu().numpy())
        test_labels.append(y)

    test_pred_embds = np.array(test_pred_embds)

    tsne_gt_data = {'x': tsne_gt, 'y': train_labels}
    tsne_pred_train_data = {'x': train_pred_embds, 'y': train_labels}
    tsne_pred_test_data = {'x': test_pred_embds, 'y': test_labels}

    tsne_gt_name = 'vis_' + save_prefix + '_e' + str(epoch_pt) + '_gt_tsne_pca.png'
    tsne_test_name = 'vis_' + save_prefix + '_e' + str(epoch_pt) + '_test_tsne_pca.png'
    tsne_train_name = 'vis_' + save_prefix + '_e' + str(epoch_pt) + '_train_tsne_pca.png'

    # print("Train dataset size for plotting: ", tsne_pred_train_data['x'].shape)
    # print("Test dataset size for plotting: ", tsne_pred_test_data['x'].shape)
    # print("TSNE dataset size for plotting: ", tsne_gt_data['x'].shape)

    plot_mappings(tsne_gt_data, tsne_gt_name)
    plot_mappings(tsne_pred_train_data, tsne_train_name)
    plot_mappings(tsne_pred_test_data, tsne_test_name)


def pretrain_epoch(model, train_dl, optimizer, criteria, e, params):
    train_loss = []
    model = model.cuda()

    for i, batch in enumerate(train_dl):
        x, embds, _ = batch
        x = x.cuda()
        embds = embds.cuda()

        embds_ = model(x)

        optimizer.zero_grad()
        loss = criteria(embds_, embds)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if (i%100 == 0):
            print("[{0}/{1}][{2}/{3}] Train loss value: {4:.3f}".format(e, params['epochs_pt'], i, len(train_dl), loss.item()))

    n_batch = len(train_dl) * params['batch_size']
    train_loss_avg = np.array(train_loss).mean()

    return train_loss, train_loss_avg


def pretraining(params):
    print("-----------------------Starting Pretraining with tsne --------------------------")
    dataset_train = read_data(params['dataset'], True)
    dataset_test = read_data(params['dataset'], False)

    # dataset_train = read_mnist(True)
    # dataset_test = read_mnist(False)

    # Sampling a small set of 6K samples from training set to compute tsne and few shot training
    train_indices = np.arange(params['tsne_data_size'])

    # Warp into Subsets and DataLoaders
    dataset_train = Subset(dataset_train, train_indices)

    tsne_embeddings = query_tsne(dataset_train, params)

    tsne_embds_dataset = tsne_dataset(dataset_train, tsne_embeddings)
    train_dl = DataLoader(tsne_embds_dataset, batch_size = params['batch_size'], shuffle=True, num_workers=0)
    
    if (params['model_type'] == 'conv'):
        print("Using convolutional model for pretraining .... ")
        model = conv_feature_extractor(params['ndi'], params['ndo'])
    else:
        print("Using MLP model for pretraining ....")
        model = feature_extractor(params['ndi'], params['ndo']) 

    optimizer = torch.optim.SGD(model.parameters(), lr = params['lr_pt'], momentum=0.9)
    criteria = nn.MSELoss()

    epoch_loss = []
    for e in range(params['epochs_pt']):
        _, loss_train = pretrain_epoch(model, train_dl, optimizer, criteria, e, params)
        epoch_loss.append(loss_train)
        print("[{}/{}] Average training loss for epoch: = {:.3f}".format(e, params['epochs_pt'], loss_train))

    # Visually evaluating the difference between predicted and gt tsne-map
    evaluate_tsne_predictions(model, dataset_train, dataset_test, tsne_embeddings, params['prefix'], params['epochs_pt'])

    # Saving checkpoint for the last training step
    save_ckpt(model, params['tsne_pt_save_path_ckpt'], e)


    # Plotting the loss value over the duration of training
    f2 = plt.figure()
    plt.plot(epoch_loss)

    title = 'pretraining_loss_' + params['prefix'] + '_e' + str(params['epochs_pt'])
    save_name = 'pretraining_loss_' + params['prefix'] + '_e' + str(params['epochs_pt']) + '.png' 

    plt.title(title)
    plt.show()
    plt.savefig(save_name)    

# Function to copy the weights from the feature extractor in the intial layers of the classifier network
def replicate_weights(model, pretrain_ckpt_path):
    ckpt = torch.load(pretrain_ckpt_path)
    print("Loading checkpoint for feature extractor from :", pretrain_ckpt_path, " for epoch: ", ckpt['epoch'])

    ckpt_weights = ckpt['model_state_dict']
    print("ckpt state dict:", ckpt_weights.keys(), len(ckpt_weights.keys()))
    print("model state dict:", model.state_dict().keys(), len(model.state_dict().keys()))

    common_dict = {}

    for k,v in ckpt_weights.items():
        if (k in model.state_dict().keys()):
            common_dict[k] = v
    
    print("common dict: ", common_dict.keys(), len(common_dict.keys()))

    # Initiliazing the model dict as the current state of model dict
    model_dict = model.state_dict()
    # Updating the model dict with the new weights from the common dict
    model_dict.update(common_dict)
    # Updating the model weights with the new updated model dict
    model.load_state_dict(model_dict)
    return model


def test_epoch(dataloader, model, criteria, optimizer, e, n_epoch):
    test_loss = []
    test_acc = []
    
    for i, batch in enumerate(dataloader):
        x,y = batch
        x = x.cuda()
        y = y.cuda()

        y_ = model(x)
        loss = criteria(y_, y)

        y_preds = torch.argmax(y_,axis=1).cpu()

        correct_preds = np.zeros(y.shape[0])
        correct_preds[y.cpu()==y_preds] = 1

        acc = np.sum(correct_preds) / y.shape[0]
        
        test_loss.append(loss.item())
        test_acc.append(acc.item())

        output = "[{}/{}][{}/{}] Testing batch loss: {:.3f}, Testing batch acc: {:.2f}".format(e, n_epoch, i, len(dataloader),
                  loss.item(), acc.item())

        if (i % 2000 == 0):
            print(output)

    test_loss_avg = np.array(test_loss).sum()/len(test_loss)
    test_acc_avg = np.array(test_acc).sum()/len(test_acc)

    return test_loss_avg, test_acc_avg


def train_epoch(dataloader, model, criteria, optimizer, e, n_epoch):
    train_loss = []
    train_acc = []
    
    for i, batch in enumerate(dataloader):
        x,y = batch
        x = x.cuda()
        y = y.cuda()
        y_ = model(x)
        
        optimizer.zero_grad()
        loss = criteria(y_, y)
        loss.backward()
        optimizer.step()

        y_preds = torch.argmax(y_,axis=1).cpu()

        correct_preds = np.zeros(y.shape[0])
        correct_preds[y.cpu()==y_preds] = 1

        acc = np.sum(correct_preds) / y.shape[0]

        # Accumulating loss
        train_loss.append(loss.item())
        train_acc.append(acc.item())

        output = "[{}/{}][{}/{}] Training batch loss: {:.3f}, Training batch acc: {:.2f}".format(e, n_epoch, i, len(dataloader),
                  loss.item(), acc.item())

        if (i%1000 == 0):
            print(output)

    train_loss_avg = np.array(train_loss).sum()/len(train_loss)
    train_acc_avg = np.array(train_acc).sum()/len(train_acc)

    return train_loss_avg, train_acc_avg

def train(params):
    print("-----------------------Starting classifier training------------------------")  
    dataset_train = read_data(params['dataset'], True)
    dataset_test = read_data(params['dataset'], False)

    # dataset_train = read_mnist(True)
    # dataset_test =  read_mnist(False)

    # Sampling a new training set for few shot learning not seen by tsne pretraining and is small sized
    total_train_indices = np.arange(params['classifier_data_size'] + params['tsne_data_size'])
    train_indices = total_train_indices[params['tsne_data_size']:]

    # Warp into Subsets and DataLoaders
    dataset_train = Subset(dataset_train, train_indices)
    
    train_dl = DataLoader(dataset_train, batch_size = params['batch_size'], shuffle=True, num_workers=0)
    test_dl = DataLoader(dataset_test, batch_size = params['batch_size'], shuffle=True, num_workers=0)

    if (params['model_type'] == 'conv'):
        print("Using conv model for classification .... ")
        model = conv_classifier_model(params['ndi'])
    else:
        print("Using MLP model for classification .... ")
        model = model = classifier_model(params['ndi'])

    # Loading weights from the pre-training of tsne
    if (params['use-pretrain']):
        model = replicate_weights(model, params['pretrain_ckpt_path'])

    model = model.cuda()

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    epoch = params['epochs']
    for e in range(0, epoch):
        train_loss, train_acc = train_epoch(train_dl, model, criteria, optimizer, e, epoch) 
        test_loss, test_acc = test_epoch(test_dl, model, criteria, optimizer, e, epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(test_loss)
        val_accs.append(test_acc)

        output1 = "[{}/{}] -------------------------------------Average Epoch Stats: Train loss: {:.3f}, Train acc: {:.2f}".format(e, epoch, train_loss, train_acc)
        output2 = "[{}/{}] -------------------------------------Average Epoch Stats: Test loss: {:.3f}, Test acc: {:.2f}".format(e, epoch, test_loss, test_acc)

        print(output1)
        print(output2) 

    # Plotting the loss value over the duration of training
    plt.plot(train_accs, label='train_acc')
    plt.plot(val_accs, label='test_acc')
    
    # Defining the names to save the training stats 
    title = 'classification_tsne_pretraining_' + str(params['use-pretrain']) + '_' + params['prefix'] + '_e' + str(params['epochs'])
    save_name = 'classification_tsne_pretraining' + str(params['use-pretrain']) + '_' + params['prefix'] + '_e' + str(params['epochs']) + '.png' 

    plt.title(title) 
    plt.legend()
    plt.savefig(save_name)    



def run_main():
    params = {}
    params['lr'] = 0.0005
    params['epochs'] = 100
    params['batch_size'] = 16  
    params['model_type'] = 'conv'

    # Pretraining params
    params['lr_pt'] = 0.0005
    params['epochs_pt'] = 30

    tsne_dim = 128
    # Dimensions of the input and the embeddings
    params['ndi'] = 32*32*3
    params['ndo'] = tsne_dim


    # Number for points to be used for tsne computation
    params['tsne_data_size'] = 50000
    params['tsne_dimensions'] = tsne_dim
    params['tsne_iters'] = 600
    params['tsne_save_path'] = 'cifar10_tsne_' + str(params['tsne_data_size']) + 's_' + str(params['tsne_dimensions']) + 'd.npy'

    # Paths for checkpoint saving
    # Save checkpoint path for the pretrained model 
    params['tsne_pt_save_path_ckpt'] = '../ckpt/cifar10_tsne_fe_mt_' + str(params['model_type']) + '_' + str(params['tsne_data_size']) + 's_' + str(params['tsne_dimensions']) 
    # Save checkpoint path for trained classifier
    params['classifier_save_path_ckpt'] = '../ckpt/cifar10_clf_tsne_pt_mt_' + str(params['model_type']) + '_' +  str(params['tsne_data_size']) + 's_' + str(params['tsne_dimensions']) 
 
    # Flag for using pre-training 
    params['use-pretrain'] = False 
    # Reading ckpt for classifier training
    params['pretrain_ckpt_path'] = '../ckpt/cifar10_tsne_fe_mt_conv_20000s_128_ep49.pth'
    params['classifier_data_size'] = 20000 

    # Dataset for trainin and evaluation
    params['dataset'] = 'cifar10'

    params['prefix'] = params['dataset'] + '_md_' + params['model_type'] + '_' + str(params['tsne_data_size']) + 's_' + str(params['tsne_dimensions']) + 'd'
    
    pretraining(params)
    # train(params)  


def main():
    run_main()
    # dataset = read_cifar(True)
    # print("dataset size: ", len(dataset))
    # dataset = read_mnist(True)
    # dataset = read_svhn('train')
    # compute_tsne(dataset, 50000, params={})   

if __name__ == "__main__":
    main()

