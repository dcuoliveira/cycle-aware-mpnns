from math import ceil
import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.CoraLoader import CoraLoader
from models.GraphSAGE import GraphSAGE
from utils.parsers import get_agg_class, get_criterion

parser = argparse.ArgumentParser()

parser.add_argument("--print_every", type=int, help="Print every n batches.", default=3)
parser.add_argument("--dataset", type=str, help="Dataset to use.", default="cora")
parser.add_argument("--dataset_path", type=str, help="Path to dataset.", default=os.path.join(os.getcwd(), "src/data/inputs/cora"))
parser.add_argument("--outputs_path", type=str, help="Output path to export results.", default=os.path.join(os.getcwd(), "src/data/outputs"))
parser.add_argument("--model_name", type=str, help="Model name.", default="GraphSAGE")
parser.add_argument("--mode", type=str, help="Mode to run in.", default="train")
parser.add_argument("--task", type=str, help="Task to perform.", default="node_classification")
parser.add_argument("--agg_class", type=str, help="Aggregator class to use.", default="MeanAggregator")
parser.add_argument("--cuda", type=bool, help="Use CUDA.", default=True)
parser.add_argument("--hidden_dims", type=list, help="Hidden dimensions.", default=[16])
parser.add_argument("--dropout", type=float, help="Dropout rate.", default=0.5)
parser.add_argument("--num_samples", type=int, help="Number of samples.", default=-1)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=16)
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=200)
parser.add_argument("--lr", type=float, help="Learning rate.", default=1e-2)
parser.add_argument("--weight_decay", type=float, help="Weight decay.", default=5e-4)
parser.add_argument("--self_loop", type=bool, help="Use self loops.", default=True)
parser.add_argument("--normalize_adj", type=bool, help="Normalize adjacency.", default=True)
parser.add_argument("--transductive", type=bool, help="Transductive learning.", default=True)

if __name__ == '__main__':

    args = parser.parse_args()

    args.outputs_path = os.path.join(args.outputs_path, args.dataset, args.model_name)
    args.num_layers =  len(args.hidden_dims) + 1

    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dataset = CoraLoader(path=args.dataset_path,
                         mode=args.mode,
                         num_layers=args.num_layers,
                         self_loop=False,
                         normalize_adj=False,
                         transductive=False)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_wrapper)
    input_dim, output_dim = dataset.get_dims()

    agg_class = get_agg_class(agg_class=args.agg_class)
    model = GraphSAGE(input_dim=input_dim,
                      hidden_dims=args.hidden_dims,
                      output_dim=output_dim, 
                      agg_class=agg_class,
                      dropout=args.dropout,
                      num_samples=args.num_samples,
                      device=device)
    model.to(device)

    criterion = get_criterion(task=args.task)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_batches = int(ceil(len(dataset) / args.batch_size))

    print('--------------------------------')
    print('Training.')

    model.train()
    for epoch in range(args.epochs):
        print('Epoch {} / {}'.format(epoch+1, args.epochs))
        running_loss = 0.0
        num_correct, num_examples = 0, 0
        for (idx, batch) in enumerate(loader):
            features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model.forward(features=features,
                                node_layers=node_layers,
                                mappings=mappings,
                                rows=rows)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                predictions = torch.max(out, dim=1)[1]
                num_correct += torch.sum(predictions == labels).item()
                num_examples += len(labels)
            if (idx + 1) % args.print_every == 0:
                running_loss /= args.print_every
                accuracy = num_correct / num_examples
                print('    Batch {} / {}: loss {}, accuracy {}'.format(
                    idx+1, num_batches, running_loss, accuracy))
                running_loss = 0.0
                num_correct, num_examples = 0, 0

    print('Finished training.')
    print('--------------------------------')

    if args.save:
        print('--------------------------------')
        model_path = os.path.join(args.outputs_path, 'trained_models', args.model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        fname =  fname = 'graphsage_agg_class_{}_hidden_dims_{}_num_samples_{}_batch_size_{}_epochs_{}_lr_{}_weight_decay_{}_transductive_{}.pth'.format(
            agg_class.name, str(args.hidden_dims), str(args.num_samples), str(args.batch_size), str(args.epochs), str(args.lr), str(args.weight_decay), str(args.transductive))
        
        path = os.path.join(model_path, fname)
        print('Saving model at {}'.format(path))
        torch.save(model.state_dict(), path)
        print('Finished saving model.')
        print('--------------------------------')

    # if config['load']:
    #     directory = os.path.join(os.path.dirname(os.getcwd()), 'trained_models')
    #     fname = utils.get_fname(config)
    #     path = os.path.join(directory, fname)
    #     model.load_state_dict(torch.load(path))

    # dataset_args = (config['task'],
    #                 config['dataset'],
    #                 config['dataset_path'],
    #                 'test',
    #                 config['num_layers'],
    #                 config['self_loop'],
    #                 config['normalize_adj'],
    #                 config['transductive'])
    # dataset = utils.get_dataset(dataset_args)
    # loader = DataLoader(dataset=dataset,
    #                     batch_size=args.batch_size,
    #                     shuffle=False,
    #                     collate_fn=dataset.collate_wrapper)
    # criterion = utils.get_criterion(config['task'])
    # num_batches = int(ceil(len(dataset) / args.batch_size``))

    # print('--------------------------------')
    # print('Testing.')

    # model.eval()
    # running_loss, total_loss = 0.0, 0.0
    # num_correct, num_examples = 0, 0
    # total_correct, total_examples = 0, 0
    # for (idx, batch) in enumerate(loader):
    #     features, node_layers, mappings, rows, labels = batch
    #     features, labels = features.to(device), labels.to(device)
    #     out = model.forward(features, node_layers, mappings, rows)
    #     loss = criterion(out, labels)
    #     running_loss += loss.item()
    #     total_loss += loss.item()
    #     predictions = torch.max(out, dim=1)[1]
    #     num_correct += torch.sum(predictions == labels).item()
    #     total_correct += torch.sum(predictions == labels).item()
    #     num_examples += len(labels)
    #     total_examples += len(labels)
    #     if (idx + 1) % args.print_every == 0:
    #         running_loss /= args.print_every
    #         accuracy = num_correct / num_examples
    #         print('    Batch {} / {}: loss {}, accuracy {}'.format(
    #             idx+1, num_batches, running_loss, accuracy))
    #         running_loss = 0.0
    #         num_correct, num_examples = 0, 0
    # total_loss /= num_batches
    # total_accuracy = total_correct / total_examples

    # print('Loss {}, accuracy {}'.format(total_loss, total_accuracy))
    # print('Finished testing.')
    # print('--------------------------------')