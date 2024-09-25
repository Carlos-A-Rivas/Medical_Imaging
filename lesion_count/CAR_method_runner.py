import subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt


def main():

    # Parameters to test
    patient_num = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    dworkin_counts = [226, 114, 212, 18, 101, 23, 223, 32, 55, 42, 132, 22, 173, 69, 193]
    dworkin_vols = []
    CAR_counts = []
    CAR_vols = []
    bmap_counts = []
    bmap_vols = []

    for i, patient in enumerate(patient_num):
        # Defines probability map path
        p_map_path = f'/home/carlos/Data/patient_pr/Patient_{patient}/Patient_{patient}_pred_lesion_mask_dropout=1_probability_map.nii.gz'
        
        # Runs Binary map program for count and volume
        bmap_create = ['python', 'create_binary_map.py', p_map_path, '--patient', patient]
        result = subprocess.run(bmap_create, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        count_vol_bmap = output_lines[-1].strip().split(',')
        bmap_count = int(count_vol_bmap[0]) #####################
        bmap_vol = float(count_vol_bmap[1]) #####################

        # Runs lesion_count to get CAR count
        CAR_count_create = ['python', 'lesion_count_v1.2.py', p_map_path, '--patient', patient]
        result = subprocess.run(CAR_count_create, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        CAR_count = int(output_lines[-1].strip())  # Capture the last line and convert to int

        # Runs lesion_growth to get CAR volume
        seed_path = f"/home/carlos/Data/proposed_outputs/lesion_centers/count1.2_P{patient}_C{CAR_count}_thr0.2connec26gammaFalsegam_val0.0.nii"
        CAR_growth_create = ['python', 'lesion_growth_v1.4.py', p_map_path, seed_path, '--patient', patient]
        result = subprocess.run(CAR_growth_create, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        CAR_vol = float(output_lines[-1].strip())

        # Runs program to get the volume of the dorkin maps
        dworkin_path = f"/home/carlos/Data/dworkin_output/dworkin_P{patient}_{dworkin_counts[i]}.nii.gz"
        get_volume = ['python', 'get_volume.py', dworkin_path]
        result = subprocess.run(get_volume, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        dworkin_vol = float(output_lines[-1].strip())

        # Add everything to their respective lists
        dworkin_vols.append(dworkin_vol)
        CAR_counts.append(CAR_count)
        CAR_vols.append(CAR_vol)
        bmap_counts.append(bmap_count)
        bmap_vols.append(bmap_vol)

        '''
        # Print the output of the external script
        print(result.stdout)
        print(result.stderr)
        '''
        print(f"Patient {patient} processing complete.")

    # Define the header and data
    header = ['Patient', 'dworkin_counts', 'dworkin_vols', 'CAR_counts', 'CAR_vols', 'bmap_counts', 'bmap_vols']
    data = zip(patient_num, dworkin_counts, dworkin_vols, CAR_counts, CAR_vols, bmap_counts, bmap_vols)
    # Write to CSV file
    with open('/home/carlos/Data/proposed_outputs/stats/patient_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    print("Data saved to patient_data.csv")


    # Vol x Vol Graph
    # Plotting the identity line
    plt.plot(bmap_vols, bmap_vols, label='Identity Line (bmap_vols)', color='black', linestyle='--')
    # Plotting dworkin_vols and CAR_vols
    plt.scatter(bmap_vols, dworkin_vols, label='dworkin_vols', color='blue')
    plt.scatter(bmap_vols, CAR_vols, label='CAR_vols', color='red')
    # Adding labels and title
    plt.xlabel('bmap_vols')
    plt.ylabel('Volumes')
    plt.title('Comparison of Volumes')
    plt.legend()
    # Save the plot as a file
    plt.savefig('/home/carlos/Data/proposed_outputs/stats/volume_comparison.png')
    print("Volume comparison plot saved as volume_comparison.png")
    plt.close()


    # Count Graph
    # Plotting counts
    plt.plot(patient_num, bmap_counts, label='bmap_counts', marker='o')
    plt.plot(patient_num, dworkin_counts, label='dworkin_counts', marker='x')
    plt.plot(patient_num, CAR_counts, label='CAR_counts', marker='s')
    # Adding labels and title
    plt.xlabel('Patient Number')
    plt.ylabel('Counts')
    plt.title('Counts Comparison')
    plt.legend()
    # Save the plot as a file
    plt.savefig('/home/carlos/Data/proposed_outputs/stats/counts_comparison.png')
    print("Counts comparison plot saved as counts_comparison.png")
    plt.close()


    print("Processing complete. Plots and data saved.")
main()