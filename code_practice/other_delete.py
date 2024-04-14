import os
import pandas as pd

# 删除指定文件夹中名称含有"other"的文件
def delete_files_with_other(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if 'other' in file:
                os.remove(os.path.join(root, file))

# 删除CSV文件中Image_filename含有"other"的行，并让下一行填充删除的行
def delete_rows_with_other(csv_file):
    df = pd.read_csv(csv_file)
    rows_to_delete = df[df['Image_filename'].str.contains('other')].index
    for row in rows_to_delete:
        if row < len(df) - 1:
            df.iloc[row] = df.iloc[row + 1]
    df = df.drop(rows_to_delete)
    df.to_csv(csv_file, index=False)

# 指定文件夹路径和CSV文件路径
image_folder = 'D:\MONAI\CGAN_Image_sup\CGAN_Image\data/all_images'
csv_file = 'D:\MONAI\CGAN_Image_sup\CGAN_Image\data\image_labels.csv'

# 删除文件夹中名称含有"other"的文件
delete_files_with_other(image_folder)

# 删除CSV文件中Image_filename含有"other"的行，并让下一行填充删除的行
delete_rows_with_other(csv_file)