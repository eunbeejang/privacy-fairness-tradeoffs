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
from sklearn.ensemble import RandomForestClassifier

def mysum(*nums):
    return reduce(lambda x, y: x+y, nums)
#pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=10_000)



def train_models(args, model, teacher_loaders, criterion, optimizer, device):
    num_teachers = args.num_teachers
    models = []
    for i in range(num_teachers):
        model = RandomForestClassifier(random_state=42, warm_start=True, n_estimators=1)
        print("========== Teacher Model {} ==========".format(i))
        for _, (cats, conts, target) in enumerate(tqdm(teacher_loaders[i])):
            X = torch.cat((cats, conts), 1)
            model.fit(X, target)
            model.n_estimators += 1
        models.append(model)
    return models


def aggregated_teacher(models, dataloader, epsilon, device):
    print("========== Teacher Aggregation ==========")
    preds = torch.torch.zeros((len(models), len(dataloader.dataset)), dtype=torch.long)
    for i, model in enumerate(models):
        for cats, conts, target in dataloader:
            X = torch.cat((cats, conts), 1)
            results = model.predict(X)
            preds[i] = torch.from_numpy(results)


    labels = np.array([]).astype(int)
    for preds_labels in np.transpose(preds):
        label_counts = np.bincount(preds_labels, minlength=1)
        beta = 1 / epsilon


        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1) # LapArgMAX
            #label_counts[i] += np.random.normal(0, 1) #GauArgMAX

        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)

    return preds.numpy(), labels


def student_loader(student_train_loader, labels):
    for i, (cats, conts, target) in enumerate(student_train_loader):
        yield (cats, conts) , torch.from_numpy(labels[i * len(cats): (i + 1) * len(cats)])

def test_student(args, student_train_loader, student_labels, student_test_loader, test_size, cat_emb_size, num_conts, device, sensitive_idx):
    student_model = RandomForestClassifier(random_state=42, warm_start=True, n_estimators=100)

    print("========== Testing Student Model ==========")
    for epoch in range(args.epochs):
        train_loader = student_loader(student_train_loader, student_labels)
        for (cats, conts) , labels in train_loader:
            X = torch.cat((cats, conts), 1)
            student_model = student_model.fit(X, labels)



            test_loss = 0
            correct = 0
            i = 0

            avg_recall = 0
            avg_precision = 0
            overall_results = []
            avg_eq_odds = 0
            avg_dem_par = 0
            avg_tpr = 0
            avg_tp = 0
            avg_tn = 0
            avg_fp = 0
            avg_fn = 0

            with torch.no_grad():
                for batch_idx, (cats, conts, target) in enumerate(student_test_loader):
                    print("target\n", sum(target))
                    i+=1
                    X = torch.cat((cats, conts), 1)
                    output = student_model.predict(X)
                    output = torch.from_numpy(output)
                    pred = (output > 0.5).float()
                    print("pred\n", sum(pred))
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    curr_datetime = datetime.now()
                    curr_hour = curr_datetime.hour
                    curr_min = curr_datetime.minute

                    pred_df = pd.DataFrame(pred.numpy())
                    pred_df.to_csv(f"pred_results/{args.run_name}_{curr_hour}-{curr_min}.csv")

                    #print(pred, np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()))
                    #correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                    #total += cats.size(0)


                    # confusion matrixç
                    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
                    avg_tn += tn
                    avg_fp += fp
                    avg_fn += fn
                    avg_tp += tp

                    # position of col for sensitive values
                    sensitive = [i[sensitive_idx].item() for i in cats]
                    cat_len = max(sensitive)

                    #exit()
                    sub_cm = []
                    # print(cat_len)
                    for j in range(cat_len+1):
                        try:
                            idx = list(locate(sensitive, lambda x: x == j))
                            sub_tar = target[idx]
                            sub_pred = pred[idx]
                            sub_tn, sub_fp, sub_fn, sub_tp = confusion_matrix(sub_tar, sub_pred).ravel()
                        except:
                            # when only one value to predict
                            temp_tar = int(sub_tar.numpy()[0])
                            temp_pred = int(sub_pred.numpy()[0])
                            # print(tar, pred)
                            if temp_tar and temp_pred:
                                sub_tn, sub_fp, sub_fn, sub_tp = 0, 0, 0, 1
                            elif temp_tar and not temp_pred:
                                sub_tn, sub_fp, sub_fn, sub_tp = 0, 0, 1, 0
                            elif not temp_tar and not temp_pred:
                                sub_tn, sub_fp, sub_fn, sub_tp = 1, 0, 0, 0
                            elif not temp_tar and temp_pred:
                                sub_tn, sub_fp, sub_fn, sub_tp = 0, 1, 0, 0
                            else:
                                sub_tn, sub_fp, sub_fn, sub_tp = 0, 0, 0, 0

                        total = mysum(sub_tn, sub_fp, sub_fn, sub_tp)
                        print("??", total)
                        sub_cm.append((sub_tn / total, sub_fp / total, sub_fn / total, sub_tp / total))

                    # Fairness metrics

                    group_metrics = MetricFrame({'precision': skm.precision_score, 'recall': skm.recall_score},
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
                    sub_results = group_metrics.overall.to_dict()
                    sub_results_by_group = group_metrics.by_group.to_dict()

                    # print("\n", group_metrics.by_group, "\n")
                    avg_precision += sub_results['precision']
                    avg_recall += sub_results['recall']
                    print("pre_rec", sub_results)
                    overall_results.append(sub_results_by_group)
                    avg_eq_odds += eq_odds
                    print("eqo", eq_odds)
                    avg_dem_par += demographic_parity
                    print("dempar", demographic_parity)
                    avg_tpr += tpr.difference(method='between_groups')
                    print("tpr", tpr.difference(method='between_groups'))

            total = mysum(avg_tn, avg_fp, avg_fn, avg_tp)
            print("!!", total)
            cm = (avg_tn / total, avg_fp / total, avg_fn / total, avg_tp / total)
            test_loss /= test_size
            accuracy = correct / test_size
            avg_loss = test_loss

            return accuracy, avg_loss, avg_precision, avg_recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm, overall_results

