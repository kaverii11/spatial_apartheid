import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def generate_simulated_data(n_samples=5000):
    """
    Generates simulated raster feature cube data for the ablation study.
    Replace this with loading your actual extracted raster features.
    """
    np.random.seed(42)
    # Features: B4 (Red), B8 (NIR), B11 (SWIR)
    B4 = np.random.rand(n_samples)
    B8 = np.random.rand(n_samples)
    B11 = np.random.rand(n_samples)
    
    # Indices
    NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
    NDBI = (B11 - B8) / (B11 + B8 + 1e-8)
    
    # GLCM Texture (simulated)
    GLCM_contrast = np.random.rand(n_samples) * 10
    GLCM_dissimilarity = np.random.rand(n_samples) * 5
    
    # Labels (0: Formal, 1: Informal)
    # Simulated relationship where Informal has higher NDBI, higher Texture contrast
    logit = 2 * NDBI + 0.5 * GLCM_contrast - 1 * NDVI + np.random.randn(n_samples)
    y = (logit > 0).astype(int)
    
    df = pd.DataFrame({
        'B4': B4, 'B8': B8, 'B11': B11,
        'NDVI': NDVI, 'NDBI': NDBI,
        'GLCM_contrast': GLCM_contrast, 'GLCM_dissimilarity': GLCM_dissimilarity,
        'label': y
    })
    return df

def perform_ablation_study(df):
    """
    Performs the machine learning ablation study on the provided dataset.
    """
    # Define feature sets
    baseline_features = ['B4', 'B8', 'B11']
    intermediate_features = baseline_features + ['NDVI', 'NDBI']
    full_features = intermediate_features + ['GLCM_contrast', 'GLCM_dissimilarity']
    
    stages = {
        'Baseline': baseline_features,
        'Intermediate': intermediate_features,
        'Full Model': full_features
    }
    
    results = []
    
    # Separate target
    y = df['label']
    
    for stage_name, features in stages.items():
        X = df[features]
        
        # Stratified 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        
        # Initialize and train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Calculate metrics (Macro average)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'Stage': stage_name,
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1-Score (Macro)': f1
        })
        
    return pd.DataFrame(results)

def plot_f1_scores(results_df):
    """
    Plots a bar chart comparing F1-scores across ablation stages.
    """
    plt.figure(figsize=(8, 6))
    
    # Stages and scores
    stages = results_df['Stage']
    f1_scores = results_df['F1-Score (Macro)']
    
    # Plotting
    bars = plt.bar(stages, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])
    
    plt.title('Random Forest Ablation Study: F1-Score Comparison', fontsize=14)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom', fontsize=11)
        
    plt.tight_layout()
    plt.savefig('task2_ablation_f1_scores.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    print("Loading/Simulating data...")
    # Replace this with: df = pd.read_csv('your_raster_features.csv')
    df = generate_simulated_data(n_samples=10000)
    
    print("Performing ablation study...")
    results_df = perform_ablation_study(df)
    
    print("\n--- Ablation Study Results ---")
    print(results_df.to_markdown(index=False))
    
    # Output to CSV
    results_df.to_csv('task2_ablation_results.csv', index=False)
    print("\nResults saved to task2_ablation_results.csv")
    
    print("Generating plot...")
    plot_f1_scores(results_df)
    print("Plot saved to task2_ablation_f1_scores.png")
