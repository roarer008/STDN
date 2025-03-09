import numpy as np
import os
import pandas as pd
import argparse
import configparser
import warnings
import datetime
import re

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def seq2instance_plus(data, num_his, num_pred):
    num_step = data.shape[0]
    num_sample = num_step - num_his - num_pred + 1
    x = []
    y = []
    for i in range(num_sample):
        x.append(data[i: i + num_his])
        y.append(data[i + num_his: i + num_his + num_pred, :, :1])
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='conf/JiNan_construct_samples.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    time_slice_size = int(data_config['time_slice_size'])
    train_ratio = float(data_config['train_ratio'])
    val_ratio = float(data_config['val_ratio'])
    test_ratio = float(data_config['test_ratio'])
    num_his = int(training_config['num_his'])
    num_pred = int(training_config['num_pred'])
    num_of_vertices = int(data_config['num_of_vertices'])


    data_file = data_config['data_file']
    files = np.load(data_file, allow_pickle=True)
    data=files['data']
    #timestamp=files['timestamp']
    print(data.shape)
    #print(timestamp)
    print("Dataset: ", data.shape, data[5, 0, :])

    # Divide the dataset first ,and construct the sample
    slices = data.shape[0]
    train_slices = int(slices * 0.6)
    val_slices = int(slices * 0.2)
    test_slices = slices - train_slices - val_slices
    train_set = data[ : train_slices]
    print(train_set.shape)
    val_set = data[train_slices : val_slices + train_slices]
    print(val_set.shape)
    test_set = data[-test_slices : ]
    print(test_set.shape)

    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    xy = {}
    te = {}
    for set_name in sets.keys():
        data_set = sets[set_name]
        X, Y = seq2instance_plus(data_set[..., :1].astype("float64"), num_his, num_pred)

        xy[set_name] = [X, Y]

        time = data_set[:, 0, -1]  # timestamp
        time = pd.to_datetime(time,unit='s')
        time = pd.DatetimeIndex(time)
        print(time[0])
        dayofweek = np.reshape(time.weekday, (-1, 1))
        print(dayofweek.shape)
        timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                    // (time_slice_size * 60)  # total seconds
        timeofday = np.reshape(timeofday, (-1, 1))
        time = np.concatenate((dayofweek, timeofday), -1)
        time = seq2instance(time, num_his, num_pred)
        te[set_name] = np.concatenate(time, 1).astype(np.int32)

    x_trains, y_trains = xy['train'][0], xy['train'][1]
    x_vals, y_vals = xy['val'][0], xy['val'][1]
    x_tests, y_tests = xy['test'][0], xy['test'][1]

    trainTEs = te['train']
    valTEs = te['val']
    testTEs = te['test']
    print("train: ", x_trains.shape, y_trains.shape)
    print("val: ", x_vals.shape, y_vals.shape)
    print("test: ", x_tests.shape, y_tests.shape)
    print("trainTE: ", trainTEs.shape)
    print("valTE: ", valTEs.shape)
    print("testTE: ", testTEs.shape)
    output_dir = data_config['output_dir']
    output_path = os.path.join(output_dir, "samples_" + str(num_his) + "_" + str(num_pred) + "_" + str(time_slice_size) + ".npz")
    print(f"save file to {output_path}")
    np.savez_compressed(
            output_path,
            train_x=x_trains, train_target=y_trains,
            val_x=x_vals, val_target=y_vals,
            test_x=x_tests, test_target=y_tests,
            trainTE=trainTEs, testTE=testTEs, valTE=valTEs)