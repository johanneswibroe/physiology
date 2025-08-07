import pandas as pd
import numpy as np

def extract_subject_averages(csv_file_path, output_file_path=None):
    """
    Extract average values for each subject: last 2.8s pre and first 2.8s post
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print("Data shape:", df.shape)
    print("Column names:", df.columns.tolist())
    
    # Identify pre and post columns
    pre_columns = [col for col in df.columns if 'pre' in col.lower() and 'time' not in col.lower()]
    post_columns = [col for col in df.columns if 'post' in col.lower() and 'time' not in col.lower()]
    
    print(f"\nFound {len(pre_columns)} pre columns: {pre_columns}")
    print(f"Found {len(post_columns)} post columns: {post_columns}")
    
    # Get time column
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    
    if time_columns:
        time_col = time_columns[0]
        time_values = df[time_col]
        print(f"Using time column: {time_col}")
        print(f"Time range: {time_values.min():.3f} to {time_values.max():.3f} seconds")
        
        # Create masks for time filtering
        pre_mask = time_values >= (time_values.max() - 2.8)
        post_mask = time_values <= 2.8
    else:
        print("No time column found, using index-based filtering (assuming 0.01s intervals)")
        # Use index-based filtering (280 rows = 2.8s at 0.01s intervals)
        n_rows = min(280, len(df))
        pre_mask = df.index >= (len(df) - n_rows)
        post_mask = df.index < n_rows
    
    print(f"Pre period rows: {pre_mask.sum()} (last 2.8s)")
    print(f"Post period rows: {post_mask.sum()} (first 2.8s)")
    
    # Create results dataframe
    results = []
    
    # Extract subject numbers from column names
    subject_numbers = []
    
    # Process pre columns
    for col in pre_columns:
        # Extract subject number from column name (assuming format like "1_pre", "2_pre", etc.)
        parts = col.split('_')
        if len(parts) >= 2:
            subject_num = parts[0]
        else:
            # Fallback: use column index
            subject_num = str(pre_columns.index(col) + 1)
        
        subject_numbers.append(subject_num)
        
        # Calculate average for last 2.8 seconds
        pre_data = df.loc[pre_mask, col].dropna()
        pre_avg = pre_data.mean() if len(pre_data) > 0 else np.nan
        
        results.append({
            'Subject': subject_num,
            'Condition': 'Pre',
            'Average_Value': pre_avg,
            'Data_Points': len(pre_data)
        })
    
    # Process post columns
    for col in post_columns:
        # Extract subject number from column name
        parts = col.split('_')
        if len(parts) >= 2:
            subject_num = parts[0]
        else:
            # Fallback: use column index
            subject_num = str(post_columns.index(col) + 1)
        
        # Calculate average for first 2.8 seconds
        post_data = df.loc[post_mask, col].dropna()
        post_avg = post_data.mean() if len(post_data) > 0 else np.nan
        
        results.append({
            'Subject': subject_num,
            'Condition': 'Post',
            'Average_Value': post_avg,
            'Data_Points': len(post_data)
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Pivot to have Pre and Post as separate columns
    pivot_df = results_df.pivot(index='Subject', columns='Condition', values='Average_Value')
    
    # Add data points information
    data_points_df = results_df.pivot(index='Subject', columns='Condition', values='Data_Points')
    pivot_df['Pre_Data_Points'] = data_points_df['Pre']
    pivot_df['Post_Data_Points'] = data_points_df['Post']
    
    # Calculate difference
    pivot_df['Difference_Post_minus_Pre'] = pivot_df['Post'] - pivot_df['Pre']
    
    # Reset index to make Subject a column
    pivot_df.reset_index(inplace=True)
    
    # Reorder columns
    column_order = ['Subject', 'Pre', 'Post', 'Difference_Post_minus_Pre', 'Pre_Data_Points', 'Post_Data_Points']
    pivot_df = pivot_df[column_order]
    
    # Sort by subject number (convert to int if possible)
    try:
        pivot_df['Subject_Num'] = pd.to_numeric(pivot_df['Subject'])
        pivot_df = pivot_df.sort_values('Subject_Num').drop('Subject_Num', axis=1)
    except:
        pivot_df = pivot_df.sort_values('Subject')
    
    print(f"\nResults summary:")
    print(pivot_df)
    
    # Save to CSV
    if output_file_path is None:
        # Create output filename based on input filename
        input_name = csv_file_path.split('/')[-1].replace('.csv', '')
        output_file_path = f"{input_name}_subject_averages.csv"
    
    pivot_df.to_csv(output_file_path, index=False)
    print(f"\nResults saved to: {output_file_path}")
    
    # Print some basic statistics
    print(f"\nBasic Statistics:")
    print(f"Number of subjects: {len(pivot_df)}")
    print(f"Pre mean: {pivot_df['Pre'].mean():.4f} ± {pivot_df['Pre'].std():.4f}")
    print(f"Post mean: {pivot_df['Post'].mean():.4f} ± {pivot_df['Post'].std():.4f}")
    print(f"Average change: {pivot_df['Difference_Post_minus_Pre'].mean():.4f} ± {pivot_df['Difference_Post_minus_Pre'].std():.4f}")
    
    return pivot_df

# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "/home/joeh/sound_physiology/r5_arclight_pre_and_post.csv"
    
    try:
        result_df = extract_subject_averages(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()