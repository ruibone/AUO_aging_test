def run_hist_cell_f(run_hist_cell, id_mapping):
    
    new_cell = run_hist_cell.merge(id_mapping.iloc[:,1:], on='sheet_id', how='left') # combine only id
    col = new_cell.columns.tolist()
    col = col[4:5]+col[1:3] # discard sheet_id, create_time
    new_cell = new_cell[col]

    cell_dummy = pd.get_dummies(new_cell, columns=['eqp_id_info', 'op_id_info'])
    cell_dummy = cell_dummy.groupby(['id']).max().reset_index()
    
    return cell_dummy

##### Data Processing #####
cell_hist1 = pd.read_csv("RunHistoryCell 20M8-21M3/runhist_cell_m8m9m10m11.csv")
cell_hist2 = pd.read_csv("RunHistoryCell 20M8-21M3/runhist_cell_m12.csv")
cell_hist3 = pd.read_csv("RunHistoryCell 20M8-21M3/runhist_cell_m1.csv")
cell_hist4 = pd.read_csv("RunHistoryCell 20M8-21M3/runhist_cell_m2m3.csv")
mapping1 = pd.read_csv("RunHistoryArray 20M8-21M3/id_mapping_m8m9m10m11.csv")
mapping2 = pd.read_csv("RunHistoryArray 20M8-21M3/id_mapping_m12.csv")
mapping3 = pd.read_csv("RunHistoryArray 20M8-21M3/id_mapping_m1.csv")
mapping4 = pd.read_csv("RunHistoryArray 20M8-21M3/id_mapping_m2m3.csv")

run_hist_cell = pd.concat([cell_hist1, cell_hist2, cell_hist3, cell_hist4])
id_mapping = pd.concat([mapping1, mapping2, mapping3, mapping4])

run_hist_cell_dummy = run_hist_cell_f(run_hist_cell, id_mapping) # 208676 rows × 579 columns

# 拿學長的資料對 run_hist_cell_dummy 做 left join 後會有 136298 rows × 87 columns
# 其中有 2698 筆資料沒有對應的 run_hist_cell_dummy