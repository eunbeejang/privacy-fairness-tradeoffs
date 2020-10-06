import argparse
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import numpy as np
from model import logistic_regression
from train import run_train
from test import run_test
from data import BankDataset, make_weights_for_balanced_classes


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
    """
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    """
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
#    device = torch.device(args.device)
   

    run_results = []

    dataset = BankDataset(args.data_root)

    # Creating data indices for training and validation splits:
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_size = int(dataset_size * (1 - args.split))
    test_size = int(dataset_size * args.split)
    num_var = dataset.X.shape[1]

    # Creating PT data samplers and loaders:
    """
    COMBINE SUBSET & WEIGHT SAMPLERS
    https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907/2
    
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    """


    print("\nDATASET SIZE:\t{}\nTRAIN SET:\t{}\nTEST SET:\t{}".format(
        dataset_size, train_size, test_size))

    train_loader = DataLoader(dataset=dataset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              drop_last = False)

    test_loader = DataLoader(dataset=dataset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             drop_last = False)


    for _ in range(args.n_runs):
#        model = logistic_regression().to(device)
        model = logistic_regression(num_var)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=len(train_loader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
            privacy_engine.attach(optimizer)
        for epoch in range(1, args.epochs + 1):
#            run_train(args, model, device, train_loader, optimizer, epoch)
            run_train(args, model, train_loader, optimizer, epoch)
#        run_results.append(run_test(args, model, device, test_loader))
        run_results.append(run_test(args, model, test_loader, dataset_size, train_size, test_size))


    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
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