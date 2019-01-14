# Data Preprocessing phase!
from GrowthCurves.growth_curves import extract_clean_wieght_data, extract_dataframe_per_id, \
    add_delta_from_start_in_hours
from Utils.dataset_utils import trigger_threshold

#data preprocessing
cleaned_df = extract_clean_wieght_data()
df_per_id = extract_dataframe_per_id(cleaned_df)
df_per_id = add_delta_from_start_in_hours(df_per_id)
df_per_id = trigger_threshold(df_per_id, 1500)
stats_dict = {}
for df in df_per_id:
    stats_dict[df.ID[0]] =