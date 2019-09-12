import os

file_directory="E://Project//UDACITY_PYTORCH//fast-style-transfer//Research//image_classification\data\train\vulgar human face"
# Function to rename multiple files
def main():
    i = 0

    for filename in os.listdir(file_directory):
        #print (filename)
        rename = file_directory+"/"+"images_"+str(i).zfill(5)+".jpg"
        file_dir = file_directory+"/"+filename
        print (filename, rename, file_dir)
        os.rename(file_dir, rename)
        i += 1
    print("Total Images "+str(i))

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()