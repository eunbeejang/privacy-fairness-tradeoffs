import numpy as np

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
from model import RegressionModel
#pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=10_000)

def train(model, train_loader, criterion, optimizer, epochs, device):

    for epoch in range(epochs):
        losses = []
        model.train()

        for _batch_idx, (cats, conts, target) in enumerate(tqdm(train_loader)):
            cats, conts, target = cats.to(device), conts.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(cats, conts).view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.10f}")





def predict(model, dataloader, device):
    outputs = torch.zeros(0, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _batch_idx, (cats, conts, target) in enumerate(tqdm(dataloader)):
            cats, conts, target = cats.to(device), conts.to(device), target.to(device)
            output = model(cats, conts).view(-1)
            pred = (output > 0.5).float()
    return pred


def train_models(args, model, teacher_loaders, criterion, optimizer, device):
    num_teachers = args.num_teachers
    epoch = args.epochs
    models = []
    for i in range(num_teachers):
        print("========== Teacher Model {} ==========".format(i))
        train(model, teacher_loaders[i], criterion, optimizer, epoch, device)
        models.append(model)
    return models


def aggregated_teacher(models, dataloader, epsilon, device):
    print("========== Teacher Aggregation ==========")
    preds = torch.torch.zeros((len(models), len(dataloader.dataset)), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader, device)
        preds[i] = results


    labels = np.array([]).astype(int)
    for preds_labels in np.transpose(preds):
        label_counts = np.bincount(preds_labels, minlength=1)
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)

        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)

    return preds.numpy(), labels


def student_loader(student_train_loader, labels):
    for i, (cats, conts, target) in enumerate(student_train_loader):
        yield (cats, conts) , torch.from_numpy(labels[i * len(cats): (i + 1) * len(cats)])

def test_student(args, student_train_loader, student_labels, student_test_loader, cat_emb_size, num_conts, device):
    student_model = RegressionModel(emb_szs=cat_emb_size,
                    n_cont=num_conts,
                    emb_drop=0.04,
                    out_sz=1,
                    szs=[1000, 500, 250],
                    drops=[0.001, 0.01, 0.01],
                    y_range=(0, 1)).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0)
    steps = 0
    running_loss = 0
    correct = 0
    print("========== Testing Student Model ==========")
    for epoch in range(args.epochs):
        student_model.train()
        train_loader = student_loader(student_train_loader, student_labels)
        for (cats, conts) , labels in train_loader:
        #for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            #cats = data[0]
            #conts = data[1]
            steps += 1

            optimizer.zero_grad()
            output = student_model(cats, conts).view(-1)
            labels = labels.to(torch.float32)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #            if steps % 50 == 0:
            test_loss = 0
            correct = 0
            total = 0
            student_model.eval()
            with torch.no_grad():
                for batch_idx, (cats, conts, labels) in enumerate(student_test_loader):
                    output = student_model(cats, conts)
                    loss += criterion(output, labels).item()
                    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
                    pred = (output > 0.5).float()

                    correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
                    total += cats.size(0)
                    #correct += pred.eq(labels.view_as(pred)).sum().item()
                    #accuracy = correct

            print("Epoch: {}/{}.. ".format(epoch+1, args.epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss / len(student_train_loader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(student_test_loader)),
                  "Accuracy: {:.3f}".format(correct / total))
            running_loss = 0
