# Some helper functions
def print_results(name, table):
    print("".join(['*' for x in range(len(name) + 1)]))
    print('{}'.format(name))
    print("".join(['*' for x in range(len(name) + 1)]))
    print("Confusion matrix:")
    print(table['confusion_matrix'])
    print("Accuracy: {}".format(table['accuracy']))
    print("Recall: {}".format(table['recall']))
    print("F-score: {}".format(table['f-score']))
    print('\n')
    
def performance_metrics(true_links, result, set_size):
    validation = {}
    validation['confusion_matrix'] = recordlinkage.confusion_matrix(true_links, 
    result, set_size)
    validation['accuracy'] = recordlinkage.accuracy(true_links, result, 
    len(features['validation']))
    validation['precision'] = recordlinkage.precision(true_links, result)
    validation['recall'] = recordlinkage.recall(true_links, result)
    validation['f-score'] = recordlinkage.fscore(true_links, result)
    return validation