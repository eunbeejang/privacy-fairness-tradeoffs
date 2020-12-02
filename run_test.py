import torch
import torch.nn as nn
from tqdm import tqdm
import fairlearn.metrics as flm
import sklearn.metrics as skm
from collections import Counter
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

torch.set_printoptions(threshold=5000)


def test(args, model, device, test_loader, test_size):

    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0
    i = 0

    avg_recall = 0
    avg_recall_by_group = {}
    avg_eq_odds = 0
    avg_dem_par = 0
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
            pred_df.to_csv(f"{args.run_name}_{curr_hour}-{curr_min}.csv")

            # plot confusion matrixç
            cm = confusion_matrix(target, pred, [1, 0])


            # position of col for sensitive values
            sensitive = [i[0].item() for i in cats]


            # Fairness metrics

            group_metrics = flm.group_summary(skm.recall_score,
                                             target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)


            eq_odds = flm.equalized_odds_difference(target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)

            demographic_parity = flm.demographic_parity_difference(target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)

            #print("\n", group_metrics.by_group, "\n")
            avg_recall += group_metrics.overall
            avg_recall_by_group = dict(Counter(avg_recall_by_group)+Counter(group_metrics.by_group))
            avg_eq_odds += eq_odds
            avg_dem_par += demographic_parity
            print(cm)

    test_loss /= test_size
    accuracy = 100.0 * correct / test_size
    avg_loss = test_loss
    recall = avg_recall/i
    avg_recall_by_group = {k: v / i for k, v in avg_recall_by_group.items()}
    avg_eq_odds = avg_eq_odds/i
    avg_dem_par = avg_dem_par/i
    return accuracy, avg_loss, recall, avg_recall_by_group, avg_eq_odds, avg_dem_par, cm

