from tad.dataset.reader import GeneralDataset
"""
The only necessary part for running MERLIN.
"""
  
if __name__ == '__main__':
    print(f"--- Loading dataset: {args.dataset} ---")
    dataset = GeneralDataset.index(args.dataset.upper())
    train_, test_, labels_ = dataset_loader_map[dataset]()
    if type(train_) is list:
        dset_name = args.dataset
        for i in range(14, len(train_)):
            train, test, labels = train_[i], test_[i], labels_[i]
            print(f' running MERLIN on SMD-{i}')
            args.dataset = dset_name + '_' + str(i)
            eval(f'run_{args.model.lower()}(test, labels, args.dataset)')
    
    else:
        train, test, labels = train_, test_, labels_ 
    
        print("--- Original TranAD/MERLIN evaluation results ---")
        eval(f'run_{args.model.lower()}(test, labels, args.dataset)')
    
