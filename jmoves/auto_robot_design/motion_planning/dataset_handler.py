import numpy as np
import pandas as pd
from tqdm import tqdm

from auto_robot_design.generator.topologies.graph_manager_2l import MutationType
from auto_robot_design.motion_planning.dataset_generator import Dataset


def calc_n_sort_df_with_ws(dataset: Dataset) -> pd.DataFrame:
    upd_df = dataset.df.assign(
        total_ws=lambda x: np.sum(
            x.values[
                :, dataset.params_size : dataset.params_size + dataset.ws_grid_size
            ],
            axis=1,
        )
    )
    sorted_df = upd_df.sort_values("total_ws", ascending=False)

    return sorted_df


def filtered_df_with_ws(df: pd.DataFrame, min_ws: int) -> pd.DataFrame:
    return df[df["total_ws"] >= min_ws]


def filtered_csv_dataset(dirpath, max_chunksize, min_ws):
    dataset = Dataset(dirpath)
    path_to_csv_non_filt = dataset.path / "dataset_0.csv"
    path_to_csv_filt = dataset.path / "dataset_filt.csv"
    for chunk in pd.read_csv(path_to_csv_non_filt, chunksize=max_chunksize):
        dataset.df = chunk
        sorted_df = calc_n_sort_df_with_ws(dataset)
        filt_df = filtered_df_with_ws(sorted_df, min_ws)
        if path_to_csv_filt.exists():
            filt_df.to_csv(path_to_csv_filt, mode="a", index_label=False,index=False, header=False)
        else:
            filt_df.to_csv(path_to_csv_filt, mode="w", index_label=False, index=False,)


def update_part_old_dataset(dataset):
        mut_ranges_name = dataset.graph_manager.mutation_ranges.keys()
        jp_names = [(name[:-2], name[-1:]) for name in  mut_ranges_name]
        jp_index_s = [(dataset.graph_manager.get_node_by_name(name[0]), name[1]) for name in jp_names] 
        
        # [ for jp in jp_s if dataset.graph_manager.generator_dict[jp].generator_info]
        scaler_list = []
        for jp, index in jp_index_s:
            mut_type = dataset.graph_manager.generator_dict[jp].mutation_type
            if mut_type == MutationType.RELATIVE_PERCENTAGE and index == "0":
                scaler_list.append(-1)
            else:
                scaler_list.append(1)
        
        scaler_array = np.array(scaler_list)
        new_df = dataset.df.apply(lambda x: np.r_[x[:dataset.params_size] * scaler_array, x[dataset.params_size:]], axis=1, raw=True)

        return new_df


def update_old_dataset(dirpath, max_chunksize, name_dataset="dataset"):
        dataset = Dataset(dirpath)
        path_to_csv_filt = dataset.path / (name_dataset + ".csv")
        path_new_dataset = dataset.path / (name_dataset + str(1) + ".csv")
        for chunk in tqdm(pd.read_csv(path_to_csv_filt, chunksize=max_chunksize)):
            dataset.df = chunk
            filt_df = update_part_old_dataset(dataset)
            if path_to_csv_filt.exists():
                filt_df.to_csv(path_new_dataset, mode="a", index_label=False, index=False, header=False)
            else:
                filt_df.to_csv(path_new_dataset, mode="w", index_label=False, index=False)



if __name__ == "__main__":
    path_func = lambda x: f"/run/media/yefim-work/Samsung_data1/top_{x}"
    for i in np.arange(0,9,1):
        dirpath = path_func(int(i))
        print(dirpath)
        update_old_dataset(dirpath, 1e5)
        # filtered_csv_dataset(dirpath, 1e5, 1700)
