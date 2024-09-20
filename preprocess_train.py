import pandas as pd 
import numpy as np

train = pd.read_csv('dataset/telemetry_for_operations_training.csv')
#valid = pd.read_csv('dataset/telemetry_for_operations_validation.csv')
label = pd.read_csv('dataset/operations_labels_training.csv')


def safe_to_datetime(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.to_datetime(date_str, format='%Y-%m-%d')

# Convert timestamp columns to datetime
train['create_dt'] = train['create_dt'].apply(safe_to_datetime)
label['start_time'] = pd.to_datetime(label['start_time'])
label['end_time'] = pd.to_datetime(label['end_time'])

# As 4 is not assigned to anyone i m assing it invalid operation_kind_id
train['operation_kind_id'] = 4

datetime_mask = train['create_dt'].notna() & (train['create_dt'].dt.time != pd.Timestamp('00:00:00').time())

# Sort both DataFrames by mdm_object_name and timestamp for efficient processing
train = train.sort_values(['mdm_object_name', 'create_dt'])
label = label.sort_values(['mdm_object_name', 'start_time'])

def assign_operation_kind_id(group):
    object_labels = label[label['mdm_object_name'] == group.name]
    
    for _, row in object_labels.iterrows():
        mask = (group['create_dt'] >= row['start_time']) & (group['create_dt'] <= row['end_time']) & datetime_mask
        group.loc[mask, 'operation_kind_id'] = row['operation_kind_id']
    
    return group

train = train.groupby('mdm_object_name').apply(assign_operation_kind_id)

print(train['operation_kind_id'].value_counts(dropna=False))

# Optional: Check the percentage of date-only entries (where operation_kind_id is still 4)
date_only_percentage = (train['operation_kind_id'] == 4).mean() * 100
print(f"Percentage of date-only entries: {date_only_percentage:.2f}%")

train.to_csv('train_with_operation_kind_id.csv', index=False)
