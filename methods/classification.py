# Basic imports
import json
from typing import Optional

# Progress bar
from tqdm import tqdm
tqdm.pandas()

# Metrics
from sklearn.metrics import f1_score

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LogClassification(nn.Module): 

    """
    Classification layer. It takes style vectors of dimension
    n_feat and do a forward pass, outputing a representation
    of dimension n_class. 

    Cross Entropy will be used for the loss, no need to
    add softmax for probabilities. 
    """

    def __init__(self, n_feat, n_class): 
        super().__init__() 
        self.cls = nn.Linear(n_feat, n_class)
        
    def forward(self, x): 
        return self.cls(x)




def train_classifier(classifier: nn.Module, 
                     dataset: list, 
                     dataset_test: list, 
                     training_json: Optional[str] = "training_measurements",
                     test_json: Optional[str] = "test_measurements", 
                     print_measurements: Optional[bool] = False) -> None: 
    """
    Training method for the classification layer. After each epoch, 
    accuracy and macro-averaged F1 score are measured and stored 
    for the training and test sets. 

    After 20 epochs, it saves two json files with all recorded measurements in
    the current path. 

    Parameters:
        classifier (nn.Module): PyTorch classifier layer to be trained. 
        dataset (list): Training set containing style vectors and its 
            corresponding integer labels. It is expected that each entry of the
            list is a tuple (style_vector, label). 
        dataset_test (list): Training set containing style vectors and its
            corresponding integer labels. It is expected that each entry of the
            list is a tuple (style_vector, label).
        training_json (str) [def. 'training_measurements']: name used for the 
            produced json file containing the training set measurements. 
        test_json (str) [def. 'test_measurements']: name used for the produced
            json file containing the test set measurements.
        print_measurements (bool) [def. False]: whether to print the accuracy and F1 score
            after each epoch.
    """

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())

    # Obtain dataloaders 
    train_dataloader, train_dataloader_eval, test_dataloader =\
        _get_dataloaders(dataset, dataset_test)

    num_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # Initialize empty dictionaries to save measurements
    train_measurements, test_measurements = {}, {}

    # Obtain full list of labels for measurements
    labels_train = [x[1] for x in dataset]
    labels_test = [x[1] for x in dataset_test]

    for epoch in range(20):
        running_loss = 0
        
        #-------- Epoch training

        # Set model to training
        classifier.train()

        # Loop over batches
        for style_vecs, labels in tqdm(train_dataloader, total=num_batches,
                                       desc=f"Epoch {epoch} train", leave=False): 

            # Set gradients to zero
            optimizer.zero_grad() 

            # Compute cross entropy loss
            style_vecs = style_vecs.float()
            scores = classifier(style_vecs)
            loss = criterion(scores, labels)

            # Backward pass and weights update
            loss.backward()
            optimizer.step()

            # Keep track of loss
            running_loss += loss.item()
            
        #-------- Measurements

        # Set model to evaluation
        classifier.eval()
        with torch.no_grad():
            total, correct = 0, 0
            pred_labels = []
            for style_vecs, label in tqdm(train_dataloader_eval, total=num_batches, 
                                          desc = f"Epoch {epoch} train test", leave=False):
                
                # Obtain predictions
                style_vecs = style_vecs.float()
                scores = classifier(style_vecs)
                _, predicted = torch.max(scores, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pred_labels += [pr.item() for pr in predicted]
            
            # Compute and save train measurements
            train_acc = 100*correct / total
            f1score = f1_score(labels_train, pred_labels, average='macro')
            f1score_train = f1score # Save for printing
            train_measurements[f"Epoch {epoch}"] = {"Accuracy" : train_acc, 
                                                    "F1_score" : f1score}
            
            total, correct = 0, 0
            pred_list = []
            for style_vecs, label in tqdm(test_dataloader, total = num_test_batches, 
                                          desc = f"Epoch {epoch} test", leave=False):

                style_vecs = style_vecs.float()
                scores = classifier(style_vecs)
                _, predicted = torch.max(scores, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pred_list += [pr.item() for pr in predicted]
            
            # Compute and save test measurements
            test_acc = 100*correct / total
            f1score = f1_score(labels_test, pred_list, average='macro')
            test_measurements[f"Epoch {epoch}"] = {"Accuracy" : test_acc, 
                                                    "F1_score" : f1score}

            if print_measurements: 
                print(f"Epoch {epoch} - Train accuracy: {train_acc:.2f} %")
                print(f"Epoch {epoch} - Train F1 score (macro avg.): {f1score_train:.2f}")
                print(f"Epoch {epoch} - Test accuracy: {test_acc:.2f} %")
                print(f"Epoch {epoch} - Test f1 score (macro avg.): {f1score:.2f}") 

        with open(f"{training_json}.json", "w") as f: 
            json.dump(train_measurements, f)
        with open(f"{test_json}.json", "w") as f: 
            json.dump(test_measurements, f)


def _get_dataloaders(dataset, dataset_test):

    train_dataloader = DataLoader(dataset,
                                batch_size = 32, 
                                shuffle = True)

    train_dataloader_eval = DataLoader(dataset,
                                batch_size = 32, 
                                shuffle = False)

    test_dataloader = DataLoader(dataset_test,
                                batch_size = 32, 
                                shuffle = False)

    return (train_dataloader, 
            train_dataloader_eval, 
            test_dataloader )                          
