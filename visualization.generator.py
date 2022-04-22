import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import seaborn as sns
import glob


def plot(img_file, df, save=True):
    background = "black"
    plt.rcParams.update({'font.size': 24,
                        'font.stretch' : "semi-condensed",
                        'font.weight' : 500,
                         "lines.color": "w",
                         "patch.edgecolor": "w",
                         "text.color": "w",
                         "axes.facecolor": background,
                         "axes.edgecolor": "white",
                         "axes.labelcolor": "w",
                         "axes.labelsize": 32,
                         "axes.labelweight": 900,
                         "xtick.color": "w",
                         "ytick.color": "w",
                         "figure.facecolor": background,
                         "figure.edgecolor": background,
                         "savefig.facecolor": background,
                         "savefig.edgecolor": background,
                         "grid.color": "w",
                         "axes.grid": False,
})
    fig, ax = plt.subplot_mosaic([['00', '01', '01'], ['10', '11', '12'], [
                                 '20', '21', '22']], figsize=(60, 25), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.subplots_adjust(top=0.900,
                        bottom=0.150,
                        left=0.142,
                        right=0.881,
                        hspace=0.3,
                        wspace=0.3)
    date = datetime.datetime.fromtimestamp(
        df['second_fixed'].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f"Date of monitoring: {date}", fontsize=48)
    ax["00"].set_title("Segmented Plant Image", fontsize=38, fontweight=700)
    ax["00"].imshow(cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB))
    ax["00"].axis('off')

    sns.lineplot(x=df.index, y=df.columns[-3], data=df,
                 linewidth=9, color='r', ax=ax["01"])
    sns.lineplot(
        x=df.index, y=df.columns[0], data=df, linewidth=4, color='b', ax=ax["10"])
    sns.lineplot(
        x=df.index, y=df.columns[1], data=df, linewidth=4, color='c', ax=ax["11"])
    sns.lineplot(
        x=df.index, y=df.columns[2], data=df, linewidth=4, color='m', ax=ax["12"])
    sns.lineplot(
        x=df.index, y=df.columns[3], data=df, linewidth=4, color='y', ax=ax["20"])
    sns.lineplot(
        x=df.index, y=df.columns[4], data=df, linewidth=4, color='gray', ax=ax["21"])
    sns.lineplot(
        x=df.index, y=df.columns[5], data=df, linewidth=4, color='orange', ax=ax["22"])

    if save:
        fig.savefig(f"output/{date}.png")
        plt.cla()
        fig.clear()
        plt.close("all")

    else:
        plt.show()


def get_time(img_file, temp_sec):
    rel_s = np.float32(img_file.split(
        '-')[-1].split('.')[0]) - temp_sec.values.astype(np.int64).min()
    rel_h = np.round(rel_s/3600, 4)
    return rel_h



if __name__ == "__main__":

    RATIO = 356.9286/765.733333 # time matching factor

    df_sensor = pd.read_csv("data/fake_data_sensor.csv")
    df_sensor['hour'] = df_sensor['time_rel_h'] * 2.1
    df_sensor = df_sensor.set_index("hour")

    df_sensor["ph"] = np.round(df_sensor["ph"]*1.1, 2)
    df_sensor = df_sensor.drop(
        ["area_cm2", "volume_cm3", "time_rel_h"], axis=1)

    df_growth = pd.read_csv("data/fake_data_growth.csv")
    df_growth['second'] = df_growth['file_name'].apply(
        lambda x: x.split('-')[-1].split('.')[0])
    temp_sec = df_growth['file_name'].apply(
        lambda x: x.split('-')[-1].split('.')[0])
    df_growth['second'] = df_growth['second'].astype('int64')
    df_growth['second_fixed'] = df_growth['second'].subtract(
        df_growth['second'].min()).astype(np.int64).mul(RATIO) + df_growth.second.min()
    
    df_growth['hour'] = df_growth['second'].subtract(
        df_growth['second'].min()).div(3600)*RATIO
    df_growth = df_growth.drop(["second", "file_name"], axis=1)
    df_growth = df_growth.set_index("hour")

    df = pd.merge(df_sensor, df_growth, how="outer",
                  left_index=True, right_index=True)
    df = df.interpolate('index',  limit_direction='both')
    df = df[~df.index.duplicated()]  # remove duplicates
    df = df.rename( columns={"area_cm2": "area (unit: cm2)"})

    img_files = sorted(glob.glob("images/*"))
    tag = cv2.imread("data/hexa_logo.png")

    for i, img_file in enumerate(img_files):

        print(f"process {i}th image, {img_file} ")
        img_file = img_file
        temp_h = get_time(img_file, temp_sec)*RATIO
        temp_df = df[df.index < temp_h]
        plot(img_file, temp_df)
