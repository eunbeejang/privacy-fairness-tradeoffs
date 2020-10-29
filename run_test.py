import torch
import torch.nn as nn
from tqdm import tqdm
torch.set_printoptions(threshold=5000)
import fairlearn.metrics as flm
import sklearn.metrics as skm
from collections import Counter
from toolz.dicttoolz import valmap
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def test(args, model, device, test_loader, test_size):
    #print("\n[[EVALUATION]]\n")
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
#            test_loss += model.BCELoss(output, target).item()
            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()


            # plot confusion matrix
            cm = confusion_matrix(target, pred, [1, 0])

            """
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, ax=ax, annot_kws={"size": 5})  # annot=True to annotate cells

            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['1', '0'])
            ax.yaxis.set_ticklabels(['1', '0'])
            plt.show()
            """
            # position of col for sensitive values
#            sensitive = [i[3].item() for i in cats]
            sensitive = [i[1].item() for i in cats]


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
    """
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            test_size,
            100.0 * correct / test_size,
        )
    )

    print(
        "\nTest set: Average fairness score:\nOverall recall: {:.4f}, \nRecall by Group: {}, \nEqualized Odds: {:.4f}, \nDemographic Parity: {:.4f} \n".format(
            avg_recall/i,
            #avg_recall_by_group,
            {k: v / i for k, v in avg_recall_by_group.items()},
            avg_eq_odds/i,
#            avg_dem_par[0]/i,
            avg_dem_par/i
        )
    )
    """
    accuracy = 100.0 * correct / test_size
    avg_loss = test_loss
    recall = avg_recall/i
    avg_recall_by_group = {k: v / i for k, v in avg_recall_by_group.items()}
    avg_eq_odds = avg_eq_odds/i
    avg_dem_par = avg_dem_par/i
    return accuracy, avg_loss, recall, avg_recall_by_group, avg_eq_odds, avg_dem_par

