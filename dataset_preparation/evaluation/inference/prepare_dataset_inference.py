from dataset_preparation import prepare_dataset_commons as pdc

# Retrieve train and test dataset
inference_5 = pdc.get_simulation('0018b475-a0f3-4a3d-81cc-362e08939614', table_simulated_jobs="third_phase_simulated_jobs")
inference_10 = pdc.get_simulation('00380d92-5f47-4943-a913-5ffb16abf71b', table_simulated_jobs="third_phase_simulated_jobs")
inference_20 = pdc.get_simulation('00e88cf7-e900-449c-89f3-3291a0eb29bb', table_simulated_jobs="third_phase_simulated_jobs")
inference_50 = pdc.get_simulation('00160917-0e13-4296-9a6f-63327a087f69', table_simulated_jobs="third_phase_simulated_jobs")
inference_100 = pdc.get_simulation('0119976f-c391-48d1-b87a-51d4417980cf', table_simulated_jobs="third_phase_simulated_jobs")
inference_250 = pdc.get_simulation('000fb3a5-4de4-41e0-8b9b-348090b400ed', table_simulated_jobs="third_phase_simulated_jobs")
inference_500 = pdc.get_simulation('00329967-84a4-4b83-bd35-b3492c69d7b5', table_simulated_jobs="third_phase_simulated_jobs")
inference_1000 = pdc.get_simulation('00125429-2001-4b19-b5e8-f88917e82ab7', table_simulated_jobs="third_phase_simulated_jobs")
inference_1500 = pdc.get_simulation('01188e04-d1c9-427c-a1f0-dc3bf644ab4a', table_simulated_jobs="third_phase_simulated_jobs")
inference_2000 = pdc.get_simulation('00201945-06ef-43c3-abd8-66a0591cd8ed', table_simulated_jobs="third_phase_simulated_jobs")
inference_10000 = pdc.get_simulation('004ff8e0-8ac0-40b5-b987-8078e10a7064', table_simulated_jobs="third_phase_simulated_jobs_extrapolation_x5")

# Save files
inference_5.to_csv("inference_5", index=False, sep=";")
inference_10.to_csv("inference_10", index=False, sep=";")
inference_20.to_csv("inference_20", index=False, sep=";")
inference_50.to_csv("inference_50", index=False, sep=";")
inference_100.to_csv("inference_100", index=False, sep=";")
inference_250.to_csv("inference_250", index=False, sep=";")
inference_500.to_csv("inference_500", index=False, sep=";")
inference_1000.to_csv("inference_1000", index=False, sep=";")
inference_1500.to_csv("inference_1500", index=False, sep=";")
inference_2000.to_csv("inference_2000", index=False, sep=";")
inference_10000.to_csv("inference_10000", index=False, sep=";")
