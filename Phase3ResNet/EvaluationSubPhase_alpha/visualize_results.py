import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_results():
    if not os.path.exists('results/results_log.csv'):
        print("No results_log.csv found. Please run the evaluation first.")
        return

    results_df = pd.read_csv('results/results_log.csv')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Accuracy of Models')
    plt.savefig('results/accuracy_comparison.png')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='F1 Score', data=results_df)
    plt.title('F1 Score of Models')
    plt.savefig('results/f1_score_comparison.png')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Recall', data=results_df)
    plt.title('Recall of Models')
    plt.savefig('results/recall_comparison.png')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Sparsity', data=results_df)
    plt.title('Sparsity of Models')
    plt.savefig('results/sparsity_comparison.png')


if __name__ == '__main__':
    visualize_results()
