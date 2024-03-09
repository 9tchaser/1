import os
import shutil

# 定义原始文件夹路径和新文件夹路径
original_folders = ["D:\MONAI\CGAN_Image_sup\CGAN_Image\data/breast_download\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023", 
                    "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/archive\Dataset_BUSI_with_GT/benign",
                      "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/archive\Dataset_BUSI_with_GT\malignant",
                      "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/archive\Dataset_BUSI_with_GT/normal"]
new_folder = "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/all_images"

# 创建新文件夹
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 遍历原始文件夹
for folder in original_folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "mask" not in file and "tumor" not in file:
                file_path = os.path.join(root, file)
                # 复制文件到新文件夹
                shutil.copy(file_path, new_folder)