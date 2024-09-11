import pandas as pd
import numpy as np
import os
from quaternions_generate import IMU_EKF  # Import the IMU_EKF class from quaternions_generate.py

# set the precision of numpy to 16 decimal places
np.set_printoptions(precision=16)

def create_greeting_orientation_file(sto_file):
    # define the content of the header
    content = """DataRate=52.000000
DataType=Quaternion
version=3
OpenSimVersion=4.1
endheader
time\ttorso_imu\thumerus_r_imu\thumerus_l_imu\tradius_r_imu\tradius_l_imu\tpelvis_imu\tfemur_r_imu\tfemur_l_imu\ttibia_r_imu\ttibia_l_imu\n"""
    
    # write the content to the file
    with open(sto_file, 'w') as file:
        file.write(content)

    print("File" + sto_file + "created successfully.")

def process_and_append_to_sto(sensor_file, sto_file, dt=1/52):
    # check if the sensor data file is empty
    if os.stat(sensor_file).st_size == 0:
        print(f"Error: The file {sensor_file} is empty.")
        return

    # try to read the sensor data file
    try:
        df = pd.read_csv(sensor_file)
        
        # check if the file has no data, only headers
        if df.shape[0] == 0:
            print(f"Error: The file {sensor_file} has no data, only headers.")
            return
        
    except pd.errors.EmptyDataError:
        print(f"Error: The file {sensor_file} is empty or has invalid content.")
        return
    
    # delete the 'relative_id' column if it exists
    if 'relative_id' in df.columns:
        df = df.drop(columns=['relative_id'])

    # open the sto file in append mode
    with open(sto_file, 'a') as f:
        # initialize the current time
        current_time = 0.0
        
        # go through each row in the dataframe
        for index, row in df.iterrows():
            # get the timestamp
            timestamp = f'{current_time:.16f}'  # format the timestamp to 16 decimal places
            current_time += dt  # increment the current time by the time step
            
            # initialize the list to store the quaternion values
            quat_values = []
            for i in range(0, len(row) - 1, 9):  # iterate over the sensor data in chunks of 9, imu data(accel, gyro, mag) for each sensor
                accel_data = row.iloc[i+1:i+4].values.astype(float)
                gyro_data = row.iloc[i+4:i+7].values.astype(float)
                mag_data = row.iloc[i+7:i+10].values.astype(float)

                # Initialize the EKF with the time step
                ekf = IMU_EKF(dt)
                ekf.predict(gyro_data)
                ekf.update(accel_data, mag_data)
                quaternion = ekf.x  # get the updated quaternion

                # format the quaternion values to 16 decimal places and join them with a comma
                quat_str = ','.join([f'{q:.16f}' for q in quaternion])
                quat_values.append(quat_str)
            
            # create the line to write to the sto file
            line = f"{timestamp}\t" + '\t'.join(quat_values) + '\n'
            
            # write the line to the sto file
            f.write(line)

if __name__ == '__main__':

    # define the input sensor data file and the output sto file
    sensor_file = './data/s24_init_pose.csv'  # input sensor data file
    sto_file = './orientation_file/s24_init_orientation.sto'  # output sto file

    # create the sto file with the header
    create_greeting_orientation_file(sto_file)  
    # process the sensor data and append the orientation data to the sto file
    process_and_append_to_sto(sensor_file, sto_file)



