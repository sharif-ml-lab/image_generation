import os

# Directory containing the CSV files
dir_name = 'GPT_IND_3'

# Check if the directory exists
if not os.path.exists(dir_name):
    print(f"The directory {dir_name} does not exist.")
else:
    # List to hold all commands
    commands = []
    
    # Iterate over all CSV files in the directory
    for filename in os.listdir(dir_name):
        if filename.endswith('.csv'):
            # Construct the base name for the output path without the '.csv' extension
            base_name = filename.rsplit('.', 1)[0]
            with open('www.txt') as io:
                if base_name not in io.read():
                    continue
            
                    
            # Format the command string
            command = f"python main.py --data image --space genai --method bing --task dalle3 --cpath GPT_IND_3/{filename} --opath GPT_IMAGE_2/{base_name}"
            
            # Append the command to the list
            commands.append(command)
    
    # Print all commands separated by '&&'
    print(' && '.join(commands))
