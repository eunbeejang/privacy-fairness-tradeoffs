import numpy as np

import torch
import torch.nn as nn
from torch import optim
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix
import fairlearn.metrics as flm
import sklearn.metrics as skm
from fairlearn.metrics import true_positive_rate
from fairlearn.metrics import MetricFrame

import pandas as pd
from tqdm import tqdm
from model import RegressionModel
from more_itertools import locate
from functools import reduce

def mysum(*nums):
    return reduce(lambda x, y: x+y, nums)
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

def test_student(args, student_train_loader, student_labels, student_test_loader, test_size, cat_emb_size, num_conts, device, sensitive_idx):
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
            #total = 0
            current_total = 0
            i = 0

            avg_recall = 0
            avg_recall_by_group = {}
            avg_eq_odds = 0
            avg_dem_par = 0
            avg_tpr = 0
            avg_tp = 0
            avg_tn = 0
            avg_fp = 0
            avg_fn = 0

            student_model.eval()
            with torch.no_grad():
                for batch_idx, (cats, conts, target) in enumerate(student_test_loader):
                    i+=1
                    output = student_model(cats, conts)
                    loss += criterion(output, target).item()
                    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
                    pred = (output > 0.5).float()

                    curr_datetime = datetime.now()
                    curr_hour = curr_datetime.hour
                    curr_min = curr_datetime.minute

                    pred_df = pd.DataFrame(pred.numpy())
                    pred_df.to_csv(f"pred_results/{args.run_name}_{curr_hour}-{curr_min}.csv")

                    #print(pred, np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()))
                    #correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                    #total += cats.size(0)

                    # confusion matrixç
                    tn, fp, fn, tp = confusion_matrix(target, pred, [1, 0]).ravel()
                    avg_tn += tn
                    avg_fp += fp
                    avg_fn += fn
                    avg_tp += tp
                    # position of col for sensitive values
                    correct += tp + tn
                    # position of col for sensitive values
                    sensitive = [i[sensitive_idx].item() for i in cats]
                    cat_len = len(sensitive)
                    sub_cm = []
                    for j in range(cat_len):
                        try:
                            idx = list(locate(sensitive, lambda x: x == j))
                            sub_tar = target[idx]
                            sub_pred = pred[idx]
                            tn, fp, fn, tp = confusion_matrix(sub_tar, sub_pred).ravel()
                        except:
                            # when only one value to predict
                            temp_tar = int(sub_tar.numpy()[0])
                            temp_pred = int(sub_pred.numpy()[0])
                            # print(tar, pred)
                            if temp_tar and temp_pred:
                                tn, fp, fn, tp = 0, 0, 0, 1
                            elif temp_tar and not temp_pred:
                                tn, fp, fn, tp = 0, 0, 1, 0
                            elif not temp_tar and not temp_pred:
                                tn, fp, fn, tp = 1, 0, 0, 0
                            elif not temp_tar and temp_pred:
                                tn, fp, fn, tp = 0, 1, 0, 0
                            else:
                                tn, fp, fn, tp = 0, 0, 0, 0

                        current_total = mysum(tn, fp, fn, tp)
                        total += current_total
                        sub_cm.append((tn / current_total, fp / current_total, fn / current_total, tp / current_total))

                    # Fairness metrics

                    group_metrics = MetricFrame(skm.recall_score,
                                                target, pred,
                                                sensitive_features=sensitive)

                    demographic_parity = flm.demographic_parity_difference(target, pred,
                                                                           sensitive_features=sensitive)

                    eq_odds = flm.equalized_odds_difference(target, pred,
                                                            sensitive_features=sensitive)

                    # metric_fns = {'true_positive_rate': true_positive_rate}

                    tpr = MetricFrame(true_positive_rate,
                                      target, pred,
                                      sensitive_features=sensitive)

                    # tpr = flm.true_positive_rate(target, pred,sample_weight=sensitive)

                    # print("\n", group_metrics.by_group, "\n")
                    avg_recall += group_metrics.overall
                    #avg_recall_by_group = dict(Counter(avg_recall_by_group) + Counter(group_metrics.by_group))
                    avg_eq_odds += eq_odds
                    avg_dem_par += demographic_parity
                    avg_tpr += tpr.difference(method='between_groups')

            total = mysum(avg_tn, avg_fp, avg_fn, avg_tp)
            cm = (avg_tn / total, avg_fp / total, avg_fn / total, avg_tp / total)

            test_loss /= total
            accuracy = correct / total
            avg_loss = test_loss
            recall = avg_recall / i
            #avg_recall_by_group = {k: v / i for k, v in avg_recall_by_group.items()}
            """
            avg_eq_odds = avg_eq_odds / i
            avg_dem_par = avg_dem_par / i
            avg_tpr = avg_tpr / i
            avg_tp = avg_tp / i
            avg_tn = avg_tn / i
            avg_fp = avg_fp / i
            avg_fn = avg_fn / i
            """
            return accuracy, avg_loss, recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm




            """
            print("Epoch: {}/{}.. ".format(epoch+1, args.epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss / len(student_train_loader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(student_test_loader)),
                  "Accuracy: {:.3f}".format(correct / total))
            running_loss = 0
            """