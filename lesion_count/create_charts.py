import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
file_path = '/home/carlos/Data/proposed_outputs/stats/patient_data.csv'
patient_data = pd.read_csv(file_path)


# Number of patients
num_patients = len(patient_data['Patient'])

# Positions of the bars on the x-axis
bar_width = 0.2
r1 = np.arange(num_patients)
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting individual bars for each count type side by side
size = (8,6)
plt.figure(figsize=size)

plt.bar(r1, patient_data['bmap_counts'], width=bar_width, label='Binary Map Count', color='blue')
plt.bar(r2, patient_data['dworkin_counts'], width=bar_width, label='Dworkin Count', color='purple')
plt.bar(r3, patient_data['CAR_counts'], width=bar_width, label='Our Count', color='green')

# Adding labels and title
plt.xlabel('Subject')
plt.ylabel('Number of Lesions')
plt.title('a) Lesion Count Comparison')
plt.xticks([r + bar_width for r in range(num_patients)], patient_data['Patient'])
plt.legend()

#plt.tight_layout()
plt.savefig('/home/carlos/Data/proposed_outputs/stats/count_chart.png')
plt.close()

#0.475 aspect ratio in poster
# Creating the plot
# size = 12,6 in presentation
# size = 8,6 in paper
plt.figure(figsize=size)

# Plotting the identity line (bmap_vols)
plt.plot(patient_data['bmap_vols'], patient_data['bmap_vols'], 'k--', label='Identity Line (Binary Mask Volume)')

# Plotting the dworkin_vols and CAR_vols
plt.scatter(patient_data['bmap_vols'], patient_data['dworkin_vols'], color='blue', label='Dworkin')
plt.scatter(patient_data['bmap_vols'], patient_data['CAR_vols'], color='red', label='Ours (Count + Growth)')

# Adding labels and title
plt.xlabel('Binary Mask Volume')
plt.ylabel('Total Identified Lesion Volume')
plt.title('b) Comparison of Volumes (mL)')
plt.legend()

#plt.tight_layout()
plt.savefig('/home/carlos/Data/proposed_outputs/stats/volume_graph.png')
plt.close()

