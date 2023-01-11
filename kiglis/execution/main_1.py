import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from KiglisLoader import KiglisLoader
from Network import Network
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

def get_loaders():
    norm=0
    x_path = '/home/hardli/python/Fraunhofer KIT/Interview/kiglis/x_data.txt'
    y_path = '/home/hardli/python/Fraunhofer KIT/Interview/kiglis/y_data.txt'
    kiglis_loader_creator = KiglisLoader(x_path, y_path, norm)
    # train_loader, val_loader, test_loader = kiglis_loader_creator.create()
    return kiglis_loader_creator.create()


if __name__ == "__main__":

    train_loader, val_loader, test_loader, y_data = get_loaders()
    device = torch.device('cuda:0')
    cpu = torch.device('cpu')
    model = Network().to(device)
    # loss will stay around 1.53, don't know why yet
    # all buffs are useless for improvment
    # adam can faster achieve loss=1.55 than sgd, which is reasonable
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    # scheduler = StepLR(optimizer,step_size=2,gamma=0.1)
    criterion = nn.L1Loss().to(device)

    model.train()
    for epoch in range(10):

        for batch_idx,(data,target) in enumerate(train_loader):
            data, target = data.to(device), target.cuda()
            out = model(data)
            loss = criterion(out,target)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                print("grad_norm: {}".format(grad_norm))
        
        test_loss = 0
        for data, target in val_loader:
            data, target = data.to(device), target.cuda()
            out = model(data)
            test_loss += criterion(out, target).item()

        test_loss /= len(val_loader.dataset)
        print('\nVAL set: Average loss: {:.4f}\n'.format(test_loss))

        # scheduler.step()

    model.eval()
    RSE_numerator = 0
    all_target = torch.zeros(0)
    all_out = torch.zeros(0).to(device)
    # RSE = sqrt(sum((pred_i-tarrget_i)**2)) / RSE_denominator
    # RSE_denominator = sqrt(sum((target_i-mean(target_all))**2))
    for data, target in test_loader:
        all_target = torch.cat((target,all_target),axis=0)
        data, target = data.to(device), target.cuda()
        out = model(data)
        all_out = torch.cat((out,all_out),axis=0)
        RSE_numerator += torch.sum((target-out)**2)
    # RSE metric
    RSE_denominator = torch.sqrt(torch.sum((all_target-torch.mean(all_target))**2))
    RSE_numerator = torch.sqrt(RSE_numerator)
    RSE_loss = RSE_numerator / RSE_denominator
    # CORR metric
    all_out = all_out.to(cpu)
    all_target = all_target.to(cpu)
    CORR_numerator = (all_out-torch.mean(all_out,axis=0))*(all_target-torch.mean(all_target,axis=0))
    CORR_numerator = torch.sum(CORR_numerator,axis=0)
    CORR_denominator = ((all_out-torch.mean(all_out,axis=0))**2)*((all_target-torch.mean(all_target,axis=0))**2)
    CORR_denominator = torch.sqrt(torch.sum(CORR_denominator,axis=0))
    CORR_loss = torch.sum(CORR_numerator / CORR_denominator)/(all_target.shape[1])
    print('\nTEST set: RSE = {:.4f}\n'.format(RSE_loss))
    print('\nTEST set: CORR = {:.4f}\n'.format(CORR_loss))
