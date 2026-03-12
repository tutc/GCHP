import argparse

import Main as experiment
import Benchmarks as benchmarks

def main(args):
    print(args)

    if args.dataset == 'cifar10':
        dataset = benchmarks.Cifar10Resnet18.CIFAR10RESNET18()
    elif args.dataset == 'core50':
        dataset = benchmarks.Core50Resnet18.CORE50RESNET18()
    elif args.dataset == 'cifar100':
        dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = args.step, step = args.step)
    else:
        dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = args.step, step = args.step)



    avg, last, n_vertices = experiment.run(dataset, args.prototype)
    print('Avg: ',avg)
    print('Last: ',last)
    print('Number of vertices: ',n_vertices)

if __name__ == "__main__":

    # Commandline arguments
    parser = argparse.ArgumentParser(description="GCF....")
    
    #parser.add_argument('--num_runs', dest='num_runs', default=1, type=int, help='Number of runs (default: %(default)s)')
    parser.add_argument('--dataset',  dest='dataset', default='cifar100', type=str, help='Dataset')
    parser.add_argument('--step',  dest='step', default=2, type=int, help='Step size')
    parser.add_argument('--backbone',  dest='backbone', default='resnet50', type=str, help='Features Extractor')
    parser.add_argument('--prototype',  dest='prototype', default=3000, type=int, help='Prototype capacity')

    args = parser.parse_args()
    main(args)
