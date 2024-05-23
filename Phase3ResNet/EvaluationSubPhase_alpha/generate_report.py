import os


def generate_report():
    html_report = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Pruning Report</title>
    </head>
    <body>
        <h1>Model Pruning Report</h1>
        <h2>Experiment Overview</h2>
        <p>This report summarizes the results of the model pruning experiment...</p>
        
        <h2>Results</h2>
        <h3>Accuracy Comparison</h3>
        <img src="accuracy_comparison.png" alt="Accuracy Comparison">
        
        <h3>F1 Score Comparison</h3>
        <img src="f1_score_comparison.png" alt="F1 Score Comparison">
        
        <h3>Recall Comparison</h3>
        <img src="recall_comparison.png" alt="Recall Comparison">
        
        <h3>Sparsity Comparison</h3>
        <img src="sparsity_comparison.png" alt="Sparsity Comparison">
        
        <h3>Confusion Matrices</h3>
        <p>Original Model</p>
        <img src="confusion_matrices/confusion_matrix_Original.png" alt="Confusion Matrix - Original">
        
        <!-- Add more confusion matrices as needed -->
    </body>
    </html>
    '''

    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/report.html', 'w') as f:
        f.write(html_report)


if __name__ == '__main__':
    generate_report()
