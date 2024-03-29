import numpy as np
import pandas as pd
import os
import random

class dtLoader:
    def __init__(self,
                 SOURCE,
                 SAMPLE_FREQ = 32,
                 STRIDE = 1,
                 WINDOW_LENGTH = 8,
                 FRONT_MARGIN = 2,
                 REAR_MARGIN = 2,
                 SENSORS=[]):

        """
        Splits the signals into windows
        Args:
            SOURCE          : path to the sensor data
            SAMPLE_FREQ     : sample frequency of the sensors
            STRIDE          : stride used for windowing (in seconds)
            WINDOW_LENGTH   : length of the required window in seconds
            FRONT_MARGIN    : Required offset before a fetal kick (in seconds)
            REAR_MARGIN     : Required offset after a fetal kick (in seconds)
            SENSORS         : List of the required sensor channels
        """

        self.SOURCE = SOURCE
        self.SAMPLE_FREQ = SAMPLE_FREQ
        self.STRIDE = STRIDE
        self.WINDOW_LENGTH = WINDOW_LENGTH
        self.FRONT_MARGIN = FRONT_MARGIN
        self.REAR_MARGIN = REAR_MARGIN
        self.SENSORS = SENSORS
        self.window_list = []
        self.kick_count_list = []

    def window(self, df):
        """
        Function to split a given recording into windows
        Args:
            df              : dataframe containing the sensor readings

        Returns:
            cum_sum         : total number of kicks in the given dataframe
        """

        n = len(df)
        print("Length of dataframe : ", n//self.SAMPLE_FREQ, " seconds")

        count = 0 # A variable to debounce the butten press
        # n -= 10 * self.SAMPLE_FREQ
        # if n <= 8 * self.SAMPLE_FREQ:
        #     return 0
        cum_sum = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            if df["state"][i] >= 3:
                if count >= 8:
                    cum_sum[i] = cum_sum[i - 1] + 1
                else:
                    cum_sum[i] = cum_sum[i - 1]
                count = 0
            # if df["state"][i] == 0:
            else:
                count += 1
                cum_sum[i] = cum_sum[i - 1]
        
        df["cum_sum"] = cum_sum

        i = 0
        case_dec_kick = 0
        case_inc_kick = 0

        while True:
            case_dec_kick = 0
            case_inc_kick = 0

            # terminate loop
            if i+(self.WINDOW_LENGTH + self.REAR_MARGIN) * self.SAMPLE_FREQ >= n:
                break

            # kick in front margin
            if cum_sum[i]!=cum_sum[i + self.FRONT_MARGIN * self.SAMPLE_FREQ]:

                # no kicks in rest of window
                if cum_sum[i + self.FRONT_MARGIN * self.SAMPLE_FREQ]==cum_sum[i + self.WINDOW_LENGTH * self.SAMPLE_FREQ]:
                    i += self.STRIDE * self.SAMPLE_FREQ
                    continue

                # kick in front margin and also in middle => decrease 1
                case_dec_kick += 1

            # kick in rear margin after window
            if cum_sum[i + self.WINDOW_LENGTH * self.SAMPLE_FREQ]!=cum_sum[i+(self.WINDOW_LENGTH + self.REAR_MARGIN) * self.SAMPLE_FREQ]:

                # no kicks in rest of window
                if cum_sum[i + self.FRONT_MARGIN * self.SAMPLE_FREQ]==cum_sum[i + self.WINDOW_LENGTH * self.SAMPLE_FREQ]:
                    i += self.STRIDE * self.SAMPLE_FREQ
                    continue

                # kick in rear margin and also in middle => increase 1
                case_inc_kick += 1

            raw_window = df.iloc[i:i + self.WINDOW_LENGTH * self.SAMPLE_FREQ, :].copy()
            n_window = self.remove_DC(raw_window)
            self.window_list.append(n_window)
            self.kick_count_list.append(cum_sum[i + self.WINDOW_LENGTH * self.SAMPLE_FREQ] - cum_sum[i + self.FRONT_MARGIN * self.SAMPLE_FREQ] + case_inc_kick - case_dec_kick)

            i += self.STRIDE * self.SAMPLE_FREQ

        return cum_sum[n -  1]

    def split_windows(self):

        """
        Read the recodings and split those into windows
        Args:
        Returns:
            counts                      : Dictionary which gives the number of windows with the given number of kicks
        """
        dir_list = os.listdir(self.SOURCE)
        print('-' * 80)
        print("These are the list of files in the source directory.")
        print(dir_list)

        print('-' * 80)

        kick_count = 0

        for file in dir_list:
            df = pd.read_csv(self.SOURCE+file)
            kick_count += self.window(df)
            print(f'{file} has been processed')



        print(f"{len(self.window_list)} windows have been generated")

        max_kicks = max(self.kick_count_list)
        print(f"Maximum number of kicks in a window : {max_kicks}")

        print('-' * 80)

        counts = {k: self.kick_count_list.count(k) for k in range(max_kicks+1)}

        for key in counts:
            print(key, " kick windows : ", counts[key])

        print('-' * 80)
        print(f'Total number of kicks : {kick_count}')


        assert(len(self.window_list) == len(self.kick_count_list))
        assert(min(self.kick_count_list) >= 0)
        return counts

    def remove_DC(self, u_window):
        """
        Removes the DC component
        Args:
            u_window                : data window
        Returns:
            temp.T                  : recordings without the DC components -> shape(sensors, amplitudes)
        """
        temp = u_window[self.SENSORS]
        mean = np.mean(temp, axis=0)
        temp -= np.mean(temp, axis=0)
        return temp.T

    def drop_zero_kicks(self, diff, path):
        """
        Balances the dataset by dropping few zero kick windows randomly
        Args:
            path                    : File path to save dropped zero kick windows
            diff                    : (No. of no kick windows) - (No. of kick windows)
        Returns:
            new_window_list         : numpy array with modified windows
            new_kicks_list          : numpy array with ground truth
        """

        os.mkdir(path)
        new_window_list = []
        new_kicks_list = []
        w_num = 0
        for i in range(len(self.kick_count_list)):
            if self.kick_count_list[i] == 0:
                if random.random() > 0.06 and diff > 0:
                    np.save(path + 'window' + str(w_num) + '.npy', self.window_list[i])
                    w_num += 1
                    diff -= 1
                else:
                    new_window_list.append(self.window_list[i])
                    new_kicks_list.append(0)
            elif self.kick_count_list[i] > 2:
                new_window_list.append(self.window_list[i])
                new_kicks_list.append(3)
            else:
                new_window_list.append(self.window_list[i])
                new_kicks_list.append(self.kick_count_list[i])
        new_window_list = np.array(new_window_list, dtype=np.float32)
        new_kicks_list = np.array(new_kicks_list, dtype=np.longlong)
        return (new_window_list, new_kicks_list)