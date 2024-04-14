import os
import pandas as pd

# 遍历文件夹中的所有图片文件
def list_image_files(folder):
    image_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image_files.append(os.path.join(root, file))
    return image_files

# 生成含有标签信息和图片名称的CSV文件
def generate_csv(image_files):
    data = []
    for file in image_files:
        filename = os.path.basename(file)
        if 'normal' in filename:
            label = 'normal'
        elif 'benign' in filename:
            label = 'benign'
        elif 'malignant' in filename:
            label = 'malignant'
        elif 'case' in filename:
            # 从"information.xlsx"文件中获取标签信息
            label = get_label_from_excel(filename)
        data.append([filename, label])

    df = pd.DataFrame(data, columns=['Image_filename', 'Classification'])
    df['Classification'] = df['Classification'].replace({'normal': 0, 'benign': 1, 'malignant': 2})
    df.to_csv('D:\MONAI\CGAN_Image_sup\CGAN_Image\data\image_labels.csv', index=False)

# 从"information.xlsx"文件中获取标签信息
def get_label_from_excel(filename):
    excel_file = "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/breast_download\BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    df = pd.read_excel(excel_file)
    row = df[df['Image_filename'] == filename]
    if not row.empty:
        label = row['Classification'].values[0]
        return label
    else:
        return None

# 指定图片文件夹路径
image_folder = "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/all_images"

# 获取所有图片文件
image_files = list_image_files(image_folder)

# 生成CSV文件
generate_csv(image_files)