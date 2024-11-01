import trimesh
import numpy as np
import nibabel as nib

def obj_to_nifti(obj_file, nii_file, voxel_size=1.0):
    # 读取 .obj 文件
    mesh = trimesh.load(obj_file)

    # 生成体素网格
    voxelized_mesh = mesh.voxelized(pitch=voxel_size)

    # 获取体素网格的占据情况
    voxel_grid = voxelized_mesh.matrix

    # 创建一个 NIfTI 图像
    nii_image = nib.Nifti1Image(voxel_grid.astype(np.uint8), affine=np.eye(4))
    print("Original bounding box:", mesh.bounds)
    print("Voxel grid shape:", voxel_grid.shape)

    # 保存 NIfTI 图像
    nib.save(nii_image, nii_file)

# 使用示例
obj_to_nifti('test1_mouse_bone_1.obj', 'tst2_bone_1.nii', voxel_size=1.0)
