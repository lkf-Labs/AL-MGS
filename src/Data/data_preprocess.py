### Assuming the current dataset's root directory only contains the img and label folders, which store images and corresponding labels.
import os
import shutil

import os
import shutil

def rename_files(img_folder, label_folder, train_file, test_file):
    # Get sorted lists of image and label files
    img_files = sorted(os.listdir(img_folder))
    label_files = sorted(os.listdir(label_folder))

    # Verify equal number of images and labels
    if len(img_files) != len(label_files):
        print("Mismatch in image/label file counts!")
        return

    # Create new directories for renamed files
    new_img_folder = os.path.join(img_folder, 'renamed')
    new_label_folder = os.path.join(label_folder, 'renamed')
    os.makedirs(new_img_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    # Create mapping between old and new filenames
    old_to_new_mapping = {}
    for i, (img_file, label_file) in enumerate(zip(img_files, label_files)):
        new_name = f"{i:03d}"  # Generate new filename (001, 002, etc.)

        # Remove extension to get base filename
        old_name_without_extension = os.path.splitext(img_file)[0]

        # Update mapping dictionary (without extensions)
        old_to_new_mapping[old_name_without_extension] = new_name

        # Move and rename files to new directories
        shutil.move(os.path.join(img_folder, img_file),
                  os.path.join(new_img_folder, new_name + '.png'))
        shutil.move(os.path.join(label_folder, label_file),
                  os.path.join(new_label_folder, new_name + '.png'))

    # Update indices in train/test files
    def update_indices(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            old_index = line.strip()
            if old_index in old_to_new_mapping:
                new_lines.append(old_to_new_mapping[old_index] + '\n')

        # Write updated indices back to file
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    update_indices(train_file)
    update_indices(test_file)

    print("File renaming and index updating completed!")

img_folder = '../datasets/MGD-1K/img'
label_folder = '../datasets/MGD-1K/label'
train_file = '../datasets/MGD-1K/train.txt'
test_file = '../datasets/MGD-1K/test.txt'


rename_files(img_folder, label_folder, train_file, test_file)


def extract_test_images_and_labels(img_folder, label_folder, test_file, output_folder):
    # Create output directories for test images/labels
    new_img_folder = os.path.join(output_folder, 'img')
    new_label_folder = os.path.join(output_folder, 'label')
    os.makedirs(new_img_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    # Read test indices from file (without extensions)
    with open(test_file, 'r') as f:
        test_indices = [line.strip() for line in f.readlines()]

    # Copy corresponding files to new directories
    for index in test_indices:
        img_file = index + '.png'
        label_file = index + '.png'

        img_file_path = os.path.join(img_folder, img_file)
        if os.path.exists(img_file_path):
            shutil.copy(img_file_path, os.path.join(new_img_folder, img_file))
        else:
            print(f"Warning: {img_file_path} not found")

        label_file_path = os.path.join(label_folder, label_file)
        if os.path.exists(label_file_path):
            shutil.copy(label_file_path, os.path.join(new_label_folder, label_file))
        else:
            print(f"Warning: {label_file_path} not found")

    print("Test image/label extraction completed!")




# Read test.txt file and convert contents to integer list
def read_and_convert_test_file(test_file):
    with open(test_file, 'r') as f:
        # Read each line, strip leading zeros and convert to integer
        test_indices = [int(line.strip()) for line in f.readlines()]
    return test_indices


def read_test_file(test_file):
    """Read test file and return a set of test indices

    Args:
        test_file: Path to the test index file

    Returns:
        Set of test indices (as strings)
    """
    with open(test_file, 'r') as f:
        # Read each line and strip newline characters
        test_numbers = {line.strip() for line in f.readlines()}  # Using set to remove duplicates
    return test_numbers


def create_train_file(test_file, train_file):
    """Create training file by excluding test indices from full range

    Args:
        test_file: Path to test index file
        train_file: Path where training indices will be written
    """
    test_numbers = read_test_file(test_file)

    # Generate all possible 3-digit numbers (000-999)
    all_numbers = {f"{i:03d}" for i in range(1000)}  # Using set for deduplication

    # Calculate training numbers (all numbers minus test numbers)
    train_numbers = all_numbers - test_numbers

    # Write training numbers to file (sorted ascending)
    with open(train_file, 'w') as f:
        for number in sorted(train_numbers):
            f.write(number + '\n')

    print(f"{train_file} created successfully!")


# File paths configuration
test_file = '../datasets/MGD-1K/test.txt'  # Test set index file
train_file = '../datasets/MGD-1K/train.txt'  # Training set index file (to be generated)

# Generate training file
create_train_file(test_file, train_file)
