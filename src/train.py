import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, prepare_features_target, split_dataset
from model import train_linear_regression, evaluate_model, get_feature_coefficients

def run_pipeline():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "train.csv")
    output_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    features = ['LotArea', 'TotRmsAbvGrd', 'GarageArea', 'YearBuilt', 'OverallQual']
    target = 'SalePrice'
    
    print("="*60)
    print("Starting House Price Prediction Training Pipeline")
    print("="*60)
    
    # 1. Load & Split Data
    df = load_data(data_path)
    X, y = prepare_features_target(df, features, target)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Model
    model = train_linear_regression(X_train, y_train)
    
    # 3. Evaluate Model
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*30)
    print("Model Evaluation Metrics")
    print("="*30)
    for metric_name, val in metrics.items():
        print(f"{metric_name:6s}: {val:15,.2f}")
    
    # 4. Coefficients Table
    coefs = get_feature_coefficients(model, features)
    print("\n" + "="*30)
    print("Feature Coefficients")
    print("="*30)
    print(coefs.to_string(index=False))
    
    # 5. Visualizations
    # Set plot styles
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Coefficient Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Coefficient", 
        y="Feature", 
        data=coefs, 
        palette="viridis",
        hue="Feature",
        legend=False
    )
    plt.title("Feature Coefficients (Linear Regression Impact)", fontsize=14, pad=15)
    plt.xlabel("Impact on SalePrice ($)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    coef_plot_path = os.path.join(output_dir, "coefficient_importance.png")
    plt.savefig(coef_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved feature coefficient plot to: {coef_plot_path}")
    
    # Figure 2: Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="teal", bins=30)
    plt.title("Distribution of Residuals (Errors)", fontsize=14, pad=15)
    plt.xlabel("Residual Value (Actual - Predicted)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    res_plot_path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(res_plot_path, dpi=300)
    plt.close()
    print(f"Saved residuals distribution plot to: {res_plot_path}")
    
    print("="*60)
    print("Training Pipeline Successfully Completed")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
