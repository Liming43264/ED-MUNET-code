import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
# from model.net import VGGNet
from dataset import LiverDataset
from tqdm import tqdm
import cv2
# from project.case.dcnn import Net
from ED import Net
# from project.case.ED import Net
# from project.case.MUNET import Net	#运行ED-MUNET需要将ED的运行结果输出给MUNET并更改MUNET的卷积维度
# from project.case.U_Net import Net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ion()
import math
import numpy as np
import os
from torch import nn
import scipy.io as scio
import random

'''确保每次训练一样'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def adjust_learning_rate(optimizer, epoch):


    if epoch <= 100:
        lr = args.lr
    elif (epoch > 100) and (epoch <= 200):
        lr = args.lr * 0.1
    elif (epoch > 200):
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(model, criterion, optimizer, scheduler, dataload, test_dataloaders, batch_size, num_epochs=1000):
    lossdata = []
    loss_test_list = []
    loss_seg_list = []
    t = []
    tt = []
    loss_best = 100
    loss_test = 0
    ie_list = []
    icc_list = []
    ie_list_test = []
    icc_list_test = []
    for epoch in range(num_epochs):
        n = 0
        n_test = 0
        # if epoch != 0:
        t.append(epoch + 1)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        test_loss = 0
        ie = 0
        ie_test = 0
        ic = 0
        ic_test = 0
        step = 0
        adjust_learning_rate(optimizer, epoch)
        model.train()
        # for x, y in dataload:
        for index, (x, y) in tqdm(enumerate(dataload), total=len(dataload), desc="Epoch {}".format(epoch), ncols=0):
            step += 1
            regularization_loss = 0
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(input=outputs, target=labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item()
            lossdata.append(str(loss.item()))
            rate = (step + 1) / len(dataload)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:}".format(int(rate * 100), a, b, loss), end="")
            output = torch.squeeze(outputs).data.cpu().numpy()
            label = torch.squeeze(y).data.cpu().numpy()
            if epoch % 1 == 0:
              for i in range(batch_size):
                ie += sum((label[i, :] - output[i, :]) ** 2) / sum((label[i, :]) ** 2)
                ic_zi = sum((label[i, :] - sum(label[i, :]) / len(label[i, :])) * (
                            output[i, :] - sum(output[i, :]) / len(output[i, :])))
                ic_mu = math.sqrt(sum((label[i, :] - sum(label[i, :]) / len(label[i, :])) ** 2) * sum(
                    (output[i, :] - sum(output[i, :]) / len(output[i, :])) ** 2))
                if ic_mu == 0:
                    n += 1
                else:
                    ic += ic_zi / ic_mu
        scheduler.step(epoch_loss / step)
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss / step))
        if loss_best >= epoch_loss / step:
            loss_best = epoch_loss / step
            torch.save(model.state_dict(), 'weights_best.pth')
        # if (epoch + 1) % 50 == 0:
        #     torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
        # if epoch != 0:
        if epoch % 1 == 0:
          tt.append(epoch + 1)
          iee = ie / (step * batch_size)
          icc = ic / (step * batch_size - n)
          ie_list.append(iee)
          icc_list.append(icc)
          ie_data = open('RIE.txt', 'w')
          for i in ie_list:
              ie_data.write(str(i) + '\n')
          ie_data.close()
          icc_data = open('ICC.txt', 'w')
          for i in icc_list:
              icc_data.write(str(i) + '\n')
          icc_data.close()
          ie_min = min(ie_list)
          icc_min = max(icc_list)
          show_ie = '[' + str(len(ie_list)) + ' ' + str(ie_list[len(ie_list) - 1]) + ']'
          show_icc = '[' + str(len(icc_list)) + ' ' + str(icc_list[len(icc_list) - 1]) + ']'
          plt.figure(num='RIE', dpi=640)
          plt.plot(tt, ie_list, 'b')
          plt.annotate(show_ie, xytext=(len(ie_list), ie_min),
                     xy=(len(ie_list), ie_list[len(ie_list) - 1]))
          plt.plot(tt, icc_list, 'g')
          plt.annotate(show_icc, xytext=(len(icc_list), icc_min),
                     xy=(len(icc_list), icc_list[len(icc_list) - 1]))
          plt.title('RIE/ICC')
          plt.tight_layout()
          plt.subplots_adjust(wspace=0, hspace=.3)
          plt.savefig('./RIE.png')
          plt.clf()

        # if epoch != 0:
        epoch_loss /= step
        loss_seg_list.append(epoch_loss)
        loss_data = open('loss.txt', 'w')
        for i in loss_seg_list:
            loss_data.write(str(i) + '\n')
        loss_data.close()
        train_loss = min(loss_seg_list)
        show_train_loss = '[' + str(len(loss_seg_list)) + ' ' + str(loss_seg_list[len(loss_seg_list) - 1]) + ']'
        plt.figure(num='loss', dpi=640)
        plt.plot(t, loss_seg_list, 'g')
        plt.annotate(show_train_loss, xytext=(len(loss_seg_list), train_loss),
                     xy=(len(loss_seg_list), loss_seg_list[len(loss_seg_list) - 1]))
        plt.title('train loss')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=.3)
        plt.savefig('./loss.png')
        plt.clf()

        model.eval()
        step = 0
        with torch.no_grad():
            for step, (x, y) in enumerate(test_dataloaders):
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(input=outputs, target=labels)
                test_loss += loss.item()
                output = torch.squeeze(outputs).data.cpu().numpy()
                label = torch.squeeze(y).data.cpu().numpy()
                if epoch % 1 == 0:
                  for i in range(batch_size):
                    ie_test += sum((label[i, :] - output[i, :]) ** 2) / sum((label[i, :]) ** 2)
                    ic_zi_test = sum((label[i, :] - sum(label[i, :]) / len(label[i, :])) * (
                            output[i, :] - sum(output[i, :]) / len(output[i, :])))
                    ic_mu_test = math.sqrt(sum((label[i, :] - sum(label[i, :]) / len(label[i, :])) ** 2) * sum(
                        (output[i, :] - sum(output[i, :]) / len(output[i, :])) ** 2))
                    if ic_mu_test == 0:
                        n_test += 1
                    else:
                        ic_test += ic_zi_test / ic_mu_test
            # if epoch != 0:
            if epoch % 1 == 0:
               iee_test = ie_test / (step * batch_size)
               icc_test = ic_test / (step * batch_size - n_test)
               ie_list_test.append(iee_test)
               icc_list_test.append(icc_test)
               ie_data_test = open('RIE_test.txt', 'w')
               for i in ie_list_test:
                   ie_data_test.write(str(i) + '\n')
               ie_data_test.close()
               icc_data_test = open('ICC_test.txt', 'w')
               for i in icc_list_test:
                   icc_data_test.write(str(i) + '\n')
               icc_data_test.close()
               ie_min_test = min(ie_list_test)
               icc_min_test = max(icc_list_test)
               show_ie_test = '[' + str(len(ie_list_test)) + ' ' + str(ie_list_test[len(ie_list_test) - 1]) + ']'
               show_icc_test = '[' + str(len(icc_list_test)) + ' ' + str(icc_list_test[len(icc_list_test) - 1]) + ']'
               plt.figure(num='RIE_test', dpi=640)
               plt.plot(tt, ie_list_test, 'b')
               plt.annotate(show_ie_test, xytext=(len(ie_list_test), ie_min_test),
                         xy=(len(ie_list_test), ie_list_test[len(ie_list_test) - 1]))
               plt.plot(tt, icc_list_test, 'g')
               plt.annotate(show_icc_test, xytext=(len(icc_list_test), icc_min_test),
                         xy=(len(icc_list_test), icc_list_test[len(icc_list_test) - 1]))
               plt.title('RIE/ICC_test')
               plt.tight_layout()
               plt.subplots_adjust(wspace=0, hspace=.3)
               plt.savefig('./RIE_test.png')
               plt.clf()

            # if epoch != 0:
            test_loss /= step
            loss_test_list.append(test_loss)
            loss_data_test = open('loss_test.txt', 'w')
            for i in loss_test_list:
                loss_data_test.write(str(i) + '\n')
            loss_data_test.close()
            test_loss = min(loss_test_list)
            show_test_loss = '[' + str(len(loss_test_list)) + ' ' + str(loss_test_list[len(loss_test_list) - 1]) + ']'

            plt.figure(num='loss_test', dpi=640)
            plt.plot(t, loss_test_list, 'g')
            plt.annotate(show_test_loss, xytext=(2, 0.023),
                         xy=(len(loss_test_list), loss_test_list[len(loss_test_list) - 1]))
            plt.plot(t, loss_seg_list, 'k')
            plt.annotate(show_train_loss, xytext=(2, 0.024),
                         xy=(len(loss_seg_list), loss_seg_list[len(loss_seg_list) - 1]))
            plt.title('test loss')
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=.3)
            plt.savefig('./loss_test.png')
            plt.clf()

    return model


# 训练模型
def train(args):
    setup_seed(150)
    model = Net(1, 636).to(device)
    batch_size = args.batch_size
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = 1e-8)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5) # 效果好 但收敛速度慢
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)  # 效果好 但收敛速度慢

    # optimizer = optim.Adadelta(model.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0)  # 效果差
    # optimizer = optim.Adagrad(model.parameters(), lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0) # 收敛慢 效果一般
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50,
                                                            verbose=True, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=20, step_size_down=None,
    #                                   mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
    #                                   cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1)
    # liver_dataset = LiverDataset(r'D:\python file\brain6.3\data\train',transform=x_transforms, target_transform=y_transforms)
    liver_dataset = LiverDataset('../data/train' , transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataset = LiverDataset('../data/test', transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    train_model(model, criterion, optimizer, scheduler, dataloaders, test_dataloaders, batch_size)



def test(args):
    n_test = 0
    ie_test = 0
    ic_test = 0
    model = Net(1, 636)
    a = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    liver_dataset = LiverDataset('../data/val', transform=x_transforms, target_transform=y_transforms)

    dataloaders = DataLoader(liver_dataset, batch_size=1, shuffle=False)
    model.eval()


    i = 0
    with torch.no_grad():
        for x, label in dataloaders:
            x = x.unsqueeze(1)
            y = model(x)
            output = torch.squeeze(y).data.cpu().numpy()
            label = torch.squeeze(label ).data.cpu().numpy()
            for j in range(1):
                # ie_test += sum((label - output) ** 2) / sum((label) ** 2)  # 计算平均
                ie_test = sum((label - output) ** 2) / sum((label) ** 2)  # 单次计算
                with open('test_rie.txt', 'a') as file:
                    file.write(str(ie_test) + '\n')
                ic_zi_test = sum((label - sum(label) / len(label)) * (
                        output - sum(output) / len(output)))
                ic_mu_test = math.sqrt(sum((label - sum(label) / len(label)) ** 2) * sum(
                    (output - sum(output) / len(output)) ** 2))
                if ic_mu_test == 0:
                    n_test += 1
                else:
                    # ic_test += math.sqrt( ic_zi_test / ic_mu_test ) # 计算平均
                    ic_test = math.sqrt( ic_zi_test / ic_mu_test ) # 单次计算
                    with open('test_icc.txt', 'a') as file:
                        file.write(str(ic_test) + '\n')

            # img = cv2.flip(output.reshape((16, 36)), 0)
            # label_img = cv2.flip(label.reshape((16, 36)), 0)
            i = i + 1
            label_path = '../predict_data/label/label_%03d.mat'%i
            save_path = '../predict_data/result/result_%03d.mat'%i
            scio.savemat(save_path, {'data': output})
            scio.savemat(label_path, {'data': label})

            # fileName = 'test_icc.txt'
            # with open (fileName, 'w') as file:
            #     file.write(str(ic_test)+ '\n')
            # fileName = 'test_rie.txt'
            # with open(fileName, 'w') as file:
            #     file.write(str(ie_test)+ '\n')

        iee_test = ie_test / (i)  
        icc_test = ic_test / (i)  
        print('RIE', iee_test)    
        print('ICC', icc_test)    



if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, default='test', help="train or test")
    parse.add_argument("--batch_size", type=int, default=64)
    parse.add_argument("--ckpt", type=str, default='./weights_best.pth', help="the path of model weight file")
    parse.add_argument('--lr', nargs='?', type=float, default=1e-4,
                       help='Learning Rate')
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
