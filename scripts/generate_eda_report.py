import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Generating EDA report...")
    with open(os.path.join(output_dir, "eda_report.md"), "w") as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        
        f.write("## 1. Data Overview\n")
        f.write(f"- **Number of rows:** {len(df)}\n")
        f.write(f"- **Number of columns:** {len(df.columns)}\n\n")
        
        f.write("## 2. Missing Values\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            f.write(missing.to_frame("Missing Count").to_markdown())
            f.write("\n\n")
        else:
            f.write("No missing values found.\n\n")
            
        f.write("## 3. Data Types\n")
        f.write(df.dtypes.to_frame("Data Type").to_markdown())
        f.write("\n\n")
        
        f.write("## 4. Descriptive Statistics (Numerical)\n")
        f.write(df.describe().T.to_markdown())
        f.write("\n\n")
        
        f.write("## 5. Descriptive Statistics (Categorical)\n")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            f.write(df[cat_cols].describe().T.to_markdown())
            f.write("\n\n")
    
    # Generate some plots
    print("Generating plots...")
    
    # Correlation matrix for numeric
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        # Handle NA values for correlation
        corr = numeric_df.dropna().corr()
        sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()

    # Target variable distribution if 'target_rpc' exists
    if 'target_rpc' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='target_rpc')
        plt.title("Distribution of target_rpc")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_rpc_dist.png"))
        plt.close()
        
    print(f"EDA successfully completed. Results saved to {output_dir}")

if __name__ == "__main__":
    input_file = "raw_data/merged_raw_data.csv"
    output_dir = "reports/eda"
    run_eda(input_file, output_dir)
