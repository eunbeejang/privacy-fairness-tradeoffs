import argparse
import torch
from opacus import PrivacyEngine
import torch.optim as optim
import numpy as np
from model import RegressionModel
from run_train import train
from run_test import test
from data import data_loader
import data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="BANK Dataset")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        default=.1,
        help="test split ratio (default: .1)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
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
        "--data-root",
        type=str,
        default="./bank-data/bank-additional-full.csv",
        help="Where BANK_DATASET is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    dataset = data_loader(args)
    train_data, test_data, cat_emb_size, num_conts = dataset.__getitem__()
    train_size, test_size = dataset.__len__()
    run_results = []

    for i in range(args.n_runs):
        model = RegressionModel(emb_szs=cat_emb_size,
                            n_cont=num_conts,
                            emb_drop=0.04,
                            out_sz=1,
                            szs=[1000, 500, 250],
                            drops=[0.001, 0.01, 0.01],
                            y_range=(0, 1)).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=train_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=False,
            )
            privacy_engine.attach(optimizer)

        if i == 0: # print model properties
            print(model, '\n')

        print("\n=== RUN # {} ====================================\n".format(i))

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_data, optimizer, epoch)
        run_results.append(test(args, model, device, test_data, test_size))


    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results), np.std(run_results)
            )
        )

    repro_str = (
        f"{model.name()}_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"./saved_outputs/run_results_{repro_str}.out")

    if args.save_model:
        torch.save(model.state_dict(), f"./saved_models/BANK_logistic_{repro_str}.model")


if __name__ == "__main__":
    main()