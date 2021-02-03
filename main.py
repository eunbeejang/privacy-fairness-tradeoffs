import argparse
import torch
from opacus import PrivacyEngine
from model import RegressionModel
from run_train import train
from run_test import test
from data import data_loader
from torch import optim
import torch.nn as nn
import numpy as np
from pate import train_models, aggregated_teacher, test_student
import shap
from toolz import curry
#from syft.frameworks.torch.dp import pate

import wandb
wandb.login()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Measuring Privacy and Fairness Trade-offs")
    parser.add_argument(
        "-rn",
        "--run-name",
        required=True,
        type=str,
        help="Define run name for logging",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="Input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4119,
        metavar="TB",
        help="Input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        help="Number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=.1,
        metavar="LR",
        help="Learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=list,
        default=[0, 3.0, 2.85, 2.6, 2.45, 2.3, 2.15, 2.0, 1.85, 1.6, 1.45, 1.3, 1.15, 1.0, 0.85, 0.6, 0.45, 0.3, 0.15],
        metavar="S",
        help="Noise multiplier (default [0, 0.1, 0.5, 1.0])",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        #default="bank",
        required=True,
        help="Specify the dataset you want to test on. (bank: bank marketing, adult: adult census)",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="./bank-data/bank-additional-full.csv",
        help="Path to train data",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="./bank-data/bank-additional.csv",
        help="Path to test data",
    )
    parser.add_argument(
        "--num-teachers",
        type=int,
        default=0,
        help="Number of PATE teacher (default=3)",
    )
    parser.add_argument(
        "--sensitive",
        type=str,
        required=True,
        help="Name of sensitive column",
    )
    args = parser.parse_args()
    device = torch.device(args.device)



#    for i in range(args.n_runs):
    for i, s in enumerate(args.sigma):
        if args.num_teachers == 0 or s == 0:
            dataset = data_loader(args, s)
            train_data, test_data = dataset.__getitem__()
            cat_emb_size, num_conts = dataset.get_input_properties()
            train_size, test_size = dataset.__len__()
            sensitive_cat_keys = dataset.getkeys()
            sensitive_idx = dataset.get_sensitive_idx()
            print(sensitive_cat_keys)
        else:
            dataset = data_loader(args, s)
            train_size, test_size = dataset.__len__()
            teacher_loaders = dataset.train_teachers()
            student_train_loader, student_test_loader = dataset.student_data()
            cat_emb_size, num_conts = dataset.get_input_properties()
            sensitive_cat_keys = dataset.getkeys()
            sensitive_idx = dataset.get_sensitive_idx()
            print(sensitive_cat_keys)

            print("!!!!!! DATA LOADED")

        #run_results = []

        wandb.init(project="privacy-fairness-2", name=args.run_name,  config={
            "run_name": args.run_name,
            "architecture": 'RegressionModel',
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "n_epoch": args.epochs,
            "learning_rate": args.lr,
            "sigma(noise)": s,
            "disable_dp": args.disable_dp,
        })
        config = wandb.config
        model = RegressionModel(emb_szs=cat_emb_size,
                            n_cont=num_conts,
                            emb_drop=0.04,
                            out_sz=1,
                            szs=[1000, 500, 250],
                            drops=[0.001, 0.01, 0.01],
                            y_range=(0, 1)).to(device)

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

        if not args.disable_dp:
            if s > 0:
                privacy_engine = PrivacyEngine(
                    model,
                    batch_size=args.batch_size,
                    sample_size=train_size,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=s,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    secure_rng=False,
                )
                privacy_engine.attach(optimizer)

        if args.num_teachers == 0 or s == 0:
            if i == 0: # print model properties
                print(model, '\n')

            print("\n=== RUN # {} ====================================\n".format(i))

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_data, criterion, optimizer, epoch, s)

            """

            batch = next(iter(train_data))
            cats, conts, _ = batch

            test_batch = next(iter(test_data))
            test_cats, test_conts, _ = test_batch

            explainer = shap.KernelExplainer(model, [cats.numpy(), conts.numpy()])
            print(explainer)
            shap_values = explainer.shap_values(cats.numpy())
            shap.plots.bar(shap_values)
            exit():q
            
            """
            accuracy, avg_loss, avg_precision, avg_recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm, overall_results = test(args, model, device, test_data, test_size, sensitive_idx)
        else:  # PATE MODEL
            print("!!!!!! ENTERED HERE")

            teacher_models = train_models(args, model, teacher_loaders, criterion, optimizer, device)
            preds, student_labels = aggregated_teacher(teacher_models, student_train_loader, s, device)

            accuracy, avg_loss, avg_precision, avg_recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm, overall_results = test_student(args, student_train_loader, student_labels, student_test_loader, test_size, cat_emb_size, num_conts,
                         device, sensitive_idx)

            """
            data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=student_labels,
                                                               noise_eps=s, delta=1e-5)
            print("Data Independent Epsilon:", data_ind_eps)
            print("Data Dependent Epsilon:", data_dep_eps)
            """
        
        
        #t = [accuracy, avg_loss, avg_precision, avg_recall, avg_eq_odds, avg_tpr, avg_dem_par, cm, sub_cm, overall_results]
        #[print(type(i)) for i in t]
        
        #print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(avg_loss,accuracy))
        result = """
===================
Test set: {}

accuracy: {:.4f}
average loss: {:.4f}
precision: {:.4f}
recall: {:.4f}
sub_pre_rec:
{}
cm:
{}
sub_cm:
{}
avg_eq_odds: {:.4f}
avg_tpr: {:.4f}
avg_dem_par: {:.4f}
""".format(args.run_name,
           accuracy,   
           avg_loss,
           avg_precision,
           avg_recall,
           overall_results,
           cm,
           sub_cm,
           avg_eq_odds,
           avg_tpr,
           avg_dem_par
           )

        # append run result
        file_path = 'out//all_results.' + args.run_name
        file_object = open(file_path, 'a+')
        file_object.write(result)
        file_object.close()
        print(result)
        log_dict = {"accuracy": accuracy,
                    "avg_loss": avg_loss,
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "avg_eq_odds": avg_eq_odds,
                    "avg_tpr": avg_tpr,
                    "avg_dem_par": avg_dem_par,
                    "tn": cm[0],
                    "fp": cm[1],
                    "fn": cm[2],
                    "tp": cm[3]
                    }
        """
        for j in avg_recall_by_group.keys():
            category = sensitive_cat_keys[j]
            value = avg_recall_by_group[j]
            log_dict[category] = value
        """
        print(log_dict)
        wandb.log(log_dict)




if __name__ == "__main__":
    main()
