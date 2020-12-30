import torch
import torch.nn as nn
from tqdm import tqdm
import fairlearn.metrics as flm
import sklearn.metrics as skm
from fairlearn.metrics import true_positive_rate
from fairlearn.metrics import MetricFrame
#from aif360.sklearn.metrics import equal_opportunity_difference
from collections import Counter
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
from more_itertools import locate
from functools import reduce

def mysum(*nums):
    return reduce(lambda x, y: x+y, nums)

torch.set_printoptions(threshold=5000)


def test(args, model, device, test_loader, test_size, sensitive_idx):

    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0
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
    with torch.no_grad():
        for cats, conts, target in tqdm(test_loader):
            i += 1
            cats, conts, target = cats.to(device), conts.to(device), target.to(device)


            output = model(cats, conts)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()

            curr_datetime = datetime.now()
            curr_hour = curr_datetime.hour
            curr_min = curr_datetime.minute

            pred_df = pd.DataFrame(pred.numpy())
            pred_df.to_csv(f"pred_results/{args.run_name}_{curr_hour}-{curr_min}.csv")

            # confusion matrixç
            tn, fp, fn, tp = confusion_matrix(target, pred, [1, 0]).ravel()
            avg_tp+=tn
            avg_fp+=fp
            avg_fn+=fn
            avg_tp+=tp


            # position of col for sensitive values
            sensitive = [i[sensitive_idx].item() for i in cats]
            cat_len = max(sensitive)
            sub_cm = []
            for i in range(cat_len):
                try:
                    idx = list(locate(sensitive, lambda x: x == i))
                    sub_tar = target[idx]
                    sub_pred = pred[idx]
                    tn, fp, fn, tp = confusion_matrix(sub_tar, sub_pred).ravel()
                except:
                    # when only one value to predict
                    temp_tar = int(sub_tar.numpy()[0])
                    temp_pred = int(sub_pred.numpy()[0])
                    #print(tar, pred)
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

                total = mysum(tn,fp,fn,tp)
                sub_cm.append((tn/total, fp/total, fn/total, tp/total))

            # Fairness metrics
            group_metrics = MetricFrame(skm.recall_score,
                                             target, pred,
                                              sensitive_features=sensitive)


            demographic_parity = flm.demographic_parity_difference(target, pred,
                                              sensitive_features=sensitive)

            eq_odds = flm.equalized_odds_difference(target, pred,
                                              sensitive_features=sensitive)

            #metric_fns = {'true_positive_rate': true_positive_rate}

            tpr = MetricFrame(true_positive_rate,
                             target, pred,
                             sensitive_features=sensitive)

            #tpr = flm.true_positive_rate(target, pred,sample_weight=sensitive)


            #print("\n", group_metrics.by_group, "\n")
            avg_recall += group_metrics.overall
            #avg_recall_by_group = dict(Counter(avg_recall_by_group)+Counter(group_metrics.by_group))
            avg_eq_odds += eq_odds
            avg_dem_par += demographic_parity
            avg_tpr += tpr.difference(method='between_groups')

    test_loss /= test_size
    accuracy = 100.0 * correct / test_size
    avg_loss = test_loss
    recall = avg_recall/i
    #avg_recall_by_group = {k: v / i for k, v in avg_recall_by_group.items()}
    avg_eq_odds = avg_eq_odds/i
    avg_dem_par = avg_dem_par/i
    avg_tpr = avg_tpr/i
    avg_tp = avg_tp/i
    avg_tn = avg_tn/i
    avg_fp = avg_fp/i
    avg_fn = avg_fn/i
    total = mysum(avg_tn, avg_fp, avg_fn, avg_tp)
    cm = (avg_tn/total, avg_fp/total, avg_fn/total, avg_tp/total)
    return accuracy, avg_loss, recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm

