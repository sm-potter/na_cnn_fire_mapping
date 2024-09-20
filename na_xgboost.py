import dask.dataframe as dd
from dask.distributed import Client
import xgboost as xgb
from dask_ml.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os

# Set up Dask client for distributed training
client = Client( local_directory='/explore/nobackup/people/spotter5/temp_dir', n_workers=4, threads_per_worker=1, processes=True, memory_limit='28GB')

# Output directory
out_path = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/xgboost'
os.makedirs(out_path, exist_ok=True)

# Read the Parquet directory lazily (without loading it all into memory)
df = dd.read_parquet('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/parquet_files/all_training_na.parquet', 
                     columns=['dNBR', 'dNDVI', 'dNDII', 'y'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['dNBR', 'dNDVI', 'dNDII']], df['y'], test_size=0.2, random_state=42)

# Convert to DaskDMatrix for XGBoost
dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

# XGBoost parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'learning_rate': 0.1,
    'max_depth': 8,
    'eval_metric': 'logloss',  # Metric for binary classification
    'tree_method': 'hist',  # Use histogram-based method
    'device': 'cuda',  # Use GPU for training
}

# Train the model with Dask
output = xgb.dask.train(client, params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

# Get the trained booster model
booster = output['booster']

# Make predictions on the test set
y_pred_proba = xgb.dask.predict(client, booster, X_test)

# Convert predicted probabilities to binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Convert Dask arrays to NumPy arrays for sklearn metrics
y_pred_np = y_pred.compute()
y_test_np = y_test.compute()

# Calculate classification metrics using sklearn
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np, average='binary')
recall = recall_score(y_test_np, y_pred_np, average='binary')
f1 = f1_score(y_test_np, y_pred_np, average='binary')

# Calculate IoU using confusion matrix
cm = confusion_matrix(y_test_np, y_pred_np)
TP = cm[1, 1]  # True Positives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# IoU = True Positives / (True Positives + False Positives + False Negatives)
IoU = TP / (TP + FP + FN)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"IoU: {IoU}")

# Save the classification metrics to a CSV file
results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU'],
    'Value': [accuracy, precision, recall, f1, IoU]
})

results.to_csv(os.path.join(out_path, 'xgboost_classification_results_batch.csv'), index=False)
print(f"Classification metrics saved to {os.path.join(out_path, 'xgboost_classification_results_batch.csv')}")
