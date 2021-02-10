import numpy as np
import scipy.stats
import sklearn.model_selection
import tensorflow as tf

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    return scipy.stats.mode(np_array)[0]

def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    """
    Create sliding windows from an ndarray

    Parameters:
    
        X (numpy-array)
            The numpy array to be windowed
        
        shift (int)
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)

        stride (int)
            stride of the window (dilation)

        offset (int)
            starting index of the first window
        
        flatten (function (array) -> (value or array) )
            the function to be applied to a window after it is extracted
            can be used with get_mode (see above) for extracting the label by majority voting
            ignored if is None

    Return:

        Windowed ndarray
            shape[0] is the number of windows
    """

    overall_window_size = (window_size - 1) * stride + 1
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1
    windows = []
    for i in range(num_windows):
        start_index = i * shift + offset
        this_window = X[start_index : start_index + overall_window_size : stride]
        if flatten is not None:
            this_window = flatten(this_window)
        windows.append(this_window)
    return np.array(windows)

def get_windows_dataset_from_user_list_format(user_datasets, window_size=400, shift=200, stride=1, verbose=0):
    """
    Create windows dataset in 'user-list' format using sliding windows

    Parameters:

        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        window_size = 400
            size of the window (output)

        shift = 200
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)

        stride = 1
            stride of the window (dilation)

        verbose = 0
            debug messages are printed if > 0

    
    Return:

        user_dataset_windowed
            Windowed version of the user_datasets
            Windows from different trials are combined into one array
            type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            windowed_sensor_values have shape (num_window, window_size, channels)
            windowed_activity_labels have shape (num_window)

            Labels are decided by majority vote
    """
    
    user_dataset_windowed = {}

    for user_id in user_datasets:
        if verbose > 0:
            print(f"Processing {user_id}")
        x = []
        y = []

        # Loop through each trail of each user
        for v,l in user_datasets[user_id]:
            v_windowed = sliding_window_np(v, window_size, shift, stride)
            
            # flatten the window by majority vote (1 value for each window)
            l_flattened = sliding_window_np(l, window_size, shift, stride, flatten=get_mode)
            if len(v_windowed) > 0:
                x.append(v_windowed)
                y.append(l_flattened)
            if verbose > 0:
                print(f"Data: {v_windowed.shape}, Labels: {l_flattened.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze())
    return user_dataset_windowed

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
    """
    Combine a windowed 'user-list' dataset into training and test sets

    Parameters:

        user_dataset_windowed
            dataset in the windowed 'user-list' format {user_id: ( windowed_sensor_values, windowed_activity_labels)}
        
        train_users
            list or set of users (corresponding to the user_id) to be used as training data

        test_users = None
            list or set of users (corresponding to the user_id) to be used as testing data
            if is None, then all users not in train_users will be treated as test users 

        verbose = 0
            debug messages are printed if > 0

    Return:
        (train_x, train_y, test_x, test_y)
            train_x, train_y
                the resulting training/test input values as a single numpy array
            test_x, test_y
                the resulting training/test labels as a single (1D) numpy array
    """
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for user_id in user_datasets_windowed:
        
        v,l = user_datasets_windowed[user_id]
        if user_id in train_users:
            if verbose > 0:
                print(f"{user_id} Train")
            train_x.append(v)
            train_y.append(l)
        elif test_users is None or user_id in test_users:
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v)
            test_y.append(l)
    

    if len(train_x) == 0:
        train_x = np.array([])
        train_y = np.array([])
    else:
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y).squeeze()
    
    if len(test_x) == 0:
        test_x = np.array([])
        test_y = np.array([])
    else:
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y).squeeze()

    return train_x, train_y, test_x, test_y

def get_mean_std_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset (channel-wise)
    from training users only

    Parameters:

        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted

    Return:
        (means, stds)
            means and stds of the particular users (channel-wise)
            shape: (num_channels)

    """
    
    mean_std_data = []
    for u in train_users:
        for data, _ in user_datasets[u]:
            mean_std_data.append(data)
    mean_std_data_combined = np.concatenate(mean_std_data)
    means = np.mean(mean_std_data_combined, axis=0)
    stds = np.std(mean_std_data_combined, axis=0)
    return (means, stds)

def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """

    return ((data - mean) / std)

def apply_label_map(y, label_map):
    """
    Apply a dictionary mapping to an array of labels
    Can be used to convert str labels to int labels

    Parameters:
        y
            1D array of labels
        label_map
            a label dictionary of (label_original -> label_new)

    Return:
        y_mapped
            1D array of mapped labels
            None values are present if there is no entry in the dictionary
    """

    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def filter_none_label(X, y):
    """
    Filter samples of the value None
    Can be used to exclude non-mapped values from apply_label_map

    Parameters:
        X
            data values

        y
            labels (1D)

    Return:
        (X_filtered, y_filtered)
            X_filtered
                filtered data values
            
            y_filtered
                filtered labels (of type int)
    """

    valid_mask = np.where(y != None)
    return (np.array(X[valid_mask]), np.array(y[valid_mask], dtype=int))

def pre_process_dataset_composite(user_datasets, label_map, output_shape, train_users, test_users, window_size, shift, normalise_dataset=True, verbose=0):
    """
    A composite function to process a dataset
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        2: Split the dataset into training and test set (see combine_windowed_dataset)
        3: Normalise the datasets (see get_mean_std_from_user_list_format)
        4: Apply the label map and filter labels (see apply_label_map, filter_none_label)
        5: One-hot encode the labels (see tf.keras.utils.to_categorical)
        6: Split the training set into training and validation sets (see sklearn.model_selection.train_test_split)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}

        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)

        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)

        train_users
            list or set of users (corresponding to the user_id) to be used as training data

        test_users
            list or set of users (corresponding to the user_id) to be used as testing data

        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)

        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)

        normalise_dataset = True
            applies Z-normalisation if True

        verbose = 0
            debug messages are printed if > 0

    
    Return:
        (np_train, np_val, np_test)
            three pairs of (X, y)
            X is a windowed set of data points
            y is an array of one-hot encoded labels
    """

    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    # Step 2
    train_x, train_y, test_x, test_y = combine_windowed_dataset(user_datasets_windowed, train_users)

    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        test_x = normalise(test_x, means, stds)

    # Step 4
    train_y_mapped = apply_label_map(train_y, label_map)
    test_y_mapped = apply_label_map(test_y, label_map)

    train_x, train_y_mapped = filter_none_label(train_x, train_y_mapped)
    test_x, test_y_mapped = filter_none_label(test_x, test_y_mapped)

    if verbose > 0:
        print("Test")
        print(np.unique(test_y, return_counts=True))
        print(np.unique(test_y_mapped, return_counts=True))
        print("-----------------")

        print("Train")
        print(np.unique(train_y, return_counts=True))
        print(np.unique(train_y_mapped, return_counts=True))
        print("-----------------")

    # Step 5
    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape)
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape)

    r = np.random.randint(100)
    assert train_y_one_hot[r].argmax() == train_y_mapped[r]
    assert test_y_one_hot[r].argmax() == test_y_mapped[r]

    # Step 6
    train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x, train_y_one_hot, test_size=0.20, random_state=42)

    if verbose > 0:
        print(train_x_split.shape)
        print(val_x_split.shape)
        print(test_x.shape)

    np_train = (train_x_split, train_y_split)
    np_val = (val_x_split, val_y_split)
    np_test = (test_x, test_y_one_hot)

    # original_np_train = np_train
    # original_np_val = np_val
    # original_np_test = np_test

    return (np_train, np_val, np_test)


def make_batches_reshape(data, batch_size):
    """
    Make a batched dataset from a windowed time-series by simple reshaping
    Note that the last batch is dropped if incomplete

    Parameters:
        data
            A 3D numpy array in the shape (num_windows, window_size, num_channels)

        batch_size
            the (maximum) size of the batches

    Returns:
        batched_data
            A 4D numpy array in the shape (num_batches, batch_size, window_size, num_channels)
    """

    max_len = (data.shape[0]) // batch_size * batch_size
    return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

def np_random_shuffle_index(length):
    """
    Get a list of randomly shuffled indices
    """
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_batched_dataset_generator(data, batch_size):
    """
    Create a data batch generator
    Note that the last batch might not be full

    Parameters:
        data
            A numpy array of data

        batch_size
            the (maximum) size of the batches

    Returns:
        generator<numpy array>
            a batch of the data with the same shape except the first dimension, which is now the batch size
    """

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

    # return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

