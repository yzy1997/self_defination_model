# merge two csv files into one according to the row index

import pandas as pd

def merge_csv_files(file1, file2, output_file):
    # read two csv files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # drop the first two columns, keep the rest
    df2_trimmed_columns = df2.iloc[:, 2:]  # 舍弃前两列，保留从第三列开始的列
    
    # find the minimum number of rows between the two files
    min_rows = min(len(df1), len(df2_trimmed_columns))
    
    # retain only the first min_rows rows from
    df1_trimmed = df1.iloc[:min_rows]
    df2_trimmed = df2_trimmed_columns.iloc[:min_rows]
    
    # merge the two dataframes side by side
    merged_df = pd.concat([df1_trimmed.reset_index(drop=True), df2_trimmed.reset_index(drop=True)], axis=1)
    
    # save the merged dataframe to a new csv file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")

# test the function
def test_merge_csv_files():
    file1 = './data/s24_a1(w)_t1_u.csv'
    file2 = './data/s24_a1(w)_t1_l.csv'
    output_file = './data/s24_a1_w_merged_file.csv'
    
    merge_csv_files(file1, file2, output_file)

def reset_relative_id_column(file, output_file):
    # read the csv file
    df = pd.read_csv(file)
    
    # check if 'relative_id' column exists
    if 'relative_id' in df.columns:
        # reset the 'relative_id' column to be the row index
        df['relative_id'] = range(len(df))
    else:
        print("Error: 'relative_id' column not found in the file.")
        return
    
    # save the reordered dataframe to a new csv file
    df.to_csv(output_file, index=False)
    print(f"CSV with reordered relative_id saved to {output_file}")

# test the function
def test_reset_relative_id_column():
    input_file = './data/filled_merged_file.csv'
    output_file = './data/reordered_merged_file.csv'
    
    reset_relative_id_column(input_file, output_file)

def fill_missing_data_for_all_rows(file, output_file):
    # read the csv file
    df = pd.read_csv(file)
    
    # find rows with missing data
    missing_data_rows = df[df.isna().any(axis=1)]
    
    if not missing_data_rows.empty:
        print(f"Rows with missing data:")
        for index, row in missing_data_rows.iterrows():
            missing_count = row.isna().sum()
            print(f"Row {index}: {missing_count} missing values")
            
            # fill missing data in the row
            for col in df.columns:
                if pd.isna(df.loc[index, col]):  # if the value is missing
                    # find the previous and next row index
                    prev_row_idx = index - 1 if index > 0 else None
                    next_row_idx = index + 1 if index < len(df) - 1 else None
                    
                    prev_val = df.loc[prev_row_idx, col] if prev_row_idx is not None else None
                    next_val = df.loc[next_row_idx, col] if next_row_idx is not None else None
                    
                    # fill the missing value with the average of the previous and next values
                    if pd.notna(prev_val) and pd.notna(next_val):
                        df.loc[index, col] = (prev_val + next_val) / 2
                    elif pd.notna(prev_val):  # only the previous row has a value
                        df.loc[index, col] = prev_val
                    elif pd.notna(next_val):  # only the next row has a value
                        df.loc[index, col] = next_val
    else:
        print("No missing data found.")
    
    # save the dataframe with missing data filled to a new csv file
    df.to_csv(output_file, index=False)
    print(f"CSV with missing data filled saved to {output_file}")

def test_fill_missing_data_for_all_rows():
    input_file = './data/merged_file.csv'
    output_file = './data/filled_merged_file.csv'
    
    fill_missing_data_for_all_rows(input_file, output_file)

def update_timestamp_with_sampling_rate(file, output_file, sampling_rate):
    # read the csv file
    df = pd.read_csv(file)
    
    # calculate the time interval between samples
    time_interval = 1 / sampling_rate
    
    # update the 'timestamp' column with the new timestamps
    df['timestamp'] = [i * time_interval for i in range(len(df))]
    
    # save the dataframe with updated timestamps to a new csv file
    df.to_csv(output_file, index=False)
    print(f"CSV with updated timestamps saved to {output_file}")

def test_update_timestamp_with_sampling_rate():
    input_file = './data/reordered_merged_file.csv'
    output_file = './data/updated_merged_file.csv'
    sampling_rate = 52
    
    update_timestamp_with_sampling_rate(input_file, output_file, sampling_rate)

def reorder_columns(file, output_file):
    # read the csv file
    df = pd.read_csv(file)
    
    # the original order of the prefix of sensor columns and the suffixes
    sensor_order = ['CHS', 'RU', 'LU', 'RF', 'LF', 'WAS', 'RT', 'LT', 'RC', 'LC']
    sensor_suffixes = ['_Acc_X', '_Acc_Y', '_Acc_Z', '_Gyro_X', '_Gyro_Y', '_Gyro_Z', '_Magn_X', '_Magn_Y', '_Magn_Z']
    
    # keep the columns that are not sensor data
    non_sensor_columns = ['relative_id', 'timestamp']
    
    # create a new column order with non-sensor columns first
    new_column_order = non_sensor_columns.copy()  # add non-sensor columns to the new column order
    
    # create the new column order by adding sensor columns in the specified order
    for sensor in sensor_order:
        for suffix in sensor_suffixes:
            new_column_order.append(f'{sensor}_IMU9{suffix}')
    
    # reorder the columns in the dataframe
    df_reordered = df[new_column_order]
    
    # save the reordered dataframe to a new csv file
    df_reordered.to_csv(output_file, index=False)
    print(f"CSV with reordered columns saved to {output_file}")

def test_reorder_columns():
    input_file = './data/updated_merged_file.csv'
    output_file = './data/rearranged_sensors.csv'
    
    reorder_columns(input_file, output_file)

if __name__ == "__main__":
    # test_merge_csv_files()
    # test_fill_missing_data_for_all_rows()
    # test_reset_relative_id_column()
    # test_update_timestamp_with_sampling_rate()
    # test_reorder_columns()
    file1 = './data/s24_a1(w)_t1_u.csv'
    file2 = './data/s24_a1(w)_t1_l.csv'
    output_file = './data/s24_a1_w_merged_file.csv'
    # merge_csv_files(file1, file2, output_file) # need manually reset the sensors MAC address to the body part
    fill_missing_data_for_all_rows(output_file, output_file)
    reset_relative_id_column(output_file, output_file)
    update_timestamp_with_sampling_rate(output_file, output_file, 52)
    reorder_columns(output_file, output_file)
    pass
