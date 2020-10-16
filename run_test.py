import torch
import torch.nn as nn
from tqdm import tqdm
torch.set_printoptions(threshold=5000)
import fairlearn.metrics as flm
import sklearn.metrics as skm
from collections import Counter
from toolz.dicttoolz import valmap



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


            # position of col for sensitive values
            sensitive = [i[4].item() for i in cats]


            # Fairness metrics

            group_metrics = flm.group_summary(skm.recall_score,
                                             target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)


            eq_odds_ratio = flm.equalized_odds_ratio(target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)

            demographic_ratio = flm.demographic_parity_ratio(target, pred,
                                              sensitive_features=sensitive,
                                              sample_weight=None)

            #print("\n", group_metrics.by_group, "\n")
            avg_recall += group_metrics.overall
            avg_recall_by_group = dict(Counter(avg_recall_by_group)+Counter(group_metrics.by_group))
            avg_eq_odds += eq_odds_ratio
            avg_dem_par += demographic_ratio




    test_loss /= test_size

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            test_size,
            100.0 * correct / test_size,
        )
    )

    print(
        """
        \nTest set: Average fairness score:\n
        Overall recall: {:.4f}, 
        Recall by Group: {}, 
        Equalized Odds Ratio: {:.4f}, 
        Demographic Parity Ratio: {:.4f} \n""".format(
            avg_recall/i,
            {k: v / i for k, v in avg_recall_by_group.items()},
            avg_eq_odds/i,
            avg_dem_par/i,
        )
    )
    return 100.0 * correct / test_size

