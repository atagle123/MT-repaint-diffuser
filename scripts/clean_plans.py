import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def delete_files_in_folder(folder_path):
    try:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it is a file (not a directory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
        
        print(f"All files in {folder_path} have been deleted successfully.")
    
    except OSError as e:
        print(f"Error: {folder_path} - {e.strerror}")


if __name__ == '__main__':

	current_dir=os.getcwd()
	exps_dirs=os.path.join(current_dir,"logs/pretrained")
	for dataset_name in os.listdir(exps_dirs):
		file_path= os.path.join(exps_dirs, dataset_name,"plans")

		if os.path.exists(file_path):
			for exp_name in os.listdir(file_path):
				exp_path=os.path.join(file_path ,exp_name)
				delete_files_in_folder(exp_path)
