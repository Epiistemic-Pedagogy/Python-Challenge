# Packages for analysis
import logging
import os.path
import sys

import matplotlib.pyplot as plt
import pandas as pd
# Packages for visuals
import seaborn as sns
import sklearn.metrics as metrics
# Pickle package
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
from trustee.report.trust import TrustReport

logger = logging.getLogger('server_logger')
logger1 = logging.getLogger('matplotlib.font_manager')
logger1.setLevel(logging.CRITICAL)


def decision_tree():
    # setup logger
    loglevel = 'info'
    numeric_level = getattr(logging, loglevel.upper(), None)
    file_name = 'log.txt'
    file_handler = logging.FileHandler(file_name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    logging.basicConfig(encoding='utf-8',
                        level=numeric_level,
                        force=True,
                        handlers=[
                            file_handler,
                            console_handler])

    model_name = 'decision_tree5'
    dataset_path = './data'
    class_list = [cls for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]
    for class_name in class_list:
        # fixed train size 1e5
        data_folder = os.path.join(dataset_path, class_name)
        x_train, x_test, y_train, y_test = build_dataset(data_folder)
        logger.info(f"class {class_name}\n"
                     f"shapes: {x_train.shape}, "
                     f"{x_test.shape}, "
                     f"{y_train.shape}, "
                     f"{y_test.shape}")

        # Fit the SVM model
        # Create Decision Tree classifer object
        clf = tree.DecisionTreeClassifier(max_depth=5)

        # Train Decision Tree Classifer
        clf = clf.fit(x_train, y_train)

        plot_cm(clf, x_test, y_test, name=model_name + '_' + class_name + '_cm')
        plot_dt(clf, name=model_name + '_' + class_name)


def build_dataset(data_folder, train_len=1e5):
    dataset_name = [name for name in os.listdir(data_folder) if 'dataset' in name][0]
    label_name = [name for name in os.listdir(data_folder) if 'labels' in name][0]
    pcap_name = [name for name in os.listdir(data_folder) if 'pcap' in name][0]

    label_type = [name for name in os.listdir(data_folder) if 'type' in name][0]
    label_type = int(label_type.split('.')[0].split('_')[-1])

    if label_type == 1:
        # type 1 means the label have header
        csv_data = pd.read_csv(os.path.join(data_folder, dataset_name), header=None)
        label_data = pd.read_csv(os.path.join(data_folder, label_name))
        x = csv_data.values
        y = label_data['x'].values
    else:
        csv_data = pd.read_csv(os.path.join(data_folder, dataset_name), header=None)
        label_data = pd.read_csv(os.path.join(data_folder, label_name), header=None)
        x = csv_data.drop(columns=[0]).values
        y = label_data[0].values

    dataset_len = len(x)
    train_size = train_len / dataset_len
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
    return x_train, x_test, y_train, y_test


def plot_roc(clf, x_test, y_test, name='default'):
    plt.figure(dpi=100)
    probs = clf.predict_proba(x_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(name)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_cm(clf, x_test, y_test, name='default', ax=None):
    plt.figure(dpi=100)
    y_pred = clf.predict(x_test)
    logger.info(metrics.classification_report(y_test, y_pred, digits=5))
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    logger.info(cm)
    if ax is None:
        ax = plt.axes()
    _ = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
    ax.set_title(name)


def plot_dt(clf, name='default'):
    plt.figure(dpi=400)
    ax = plt.axes()
    ax.set_title(name)
    tree.plot_tree(clf)
    plt.show()


def trust_report(clf, x_train, x_test, y_train, y_test):
    report = TrustReport(
        clf,
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        max_iter=5,
        num_pruning_iter=5,
        train_size=0.7,
        trustee_num_iter=5,
        trustee_num_stability_iter=5,
        trustee_sample_size=0.3,
        analyze_branches=True,
        analyze_stability=True,
        top_k=10,
        verbose=True,
        is_classify=True,
    )
    logger.info(report)
    report.save('result')


if __name__ == '__main__':
    decision_tree()
    exit(0)
