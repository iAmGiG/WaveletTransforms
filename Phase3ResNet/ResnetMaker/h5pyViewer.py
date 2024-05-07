import h5py

def print_structure(name, obj):
    """ Recursively prints the structures and shapes within an HDF5 file. """
    if isinstance(obj, h5py.Dataset):  # Check if the object is a dataset
        print(f"{name}: {obj.shape}")  # Print the shape of the dataset
    elif isinstance(obj, h5py.Group):  # Check if the object is a group
        print(f"{name}: Group")  # Indicate that this is a group
        # If it's a group, iterate through the items in the group and print
        for key in obj.keys():
            print_structure(f"{name}/{key}", obj[key])

def view_weights(h5_path):
    """ View the contents of an h5 file. """
    with h5py.File(h5_path, 'r') as file:
        print("Layers and their corresponding shapes in the model:")
        file.visititems(print_structure)  # Visit each item in the file recursively

if __name__ == '__main__':
    h5_path = 'tf_model.h5'  # Update this path to the location of your .h5 file
    view_weights(h5_path)
