import trimesh
import numpy as np
import nibabel as nib

def obj_to_nifti(obj_file, nii_file, voxel_size=1.0):
    # 读取 .obj 文件
    mesh = trimesh.load(obj_file)

    # 获取模型的边界
    bounds = mesh.bounds

    # 计算需要的体素网格的大小
    grid_shape = np.ceil((bounds[1] - bounds[0]) / voxel_size).astype(int)

    # 创建一个足够大的体素网格
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

    # 生成体素网格
    voxelized_mesh = mesh.voxelized(pitch=voxel_size)

    # 获取体素网格的占据情况
    occupied_voxels = voxelized_mesh.matrix

    # 计算体素网格的原点
    origin = np.floor(bounds[0] / voxel_size).astype(int)

    # 将占据的体素填充到大的体素网格中
    min_corner = origin
    max_corner = min_corner + occupied_voxels.shape

    # 确保 min_corner 和 max_corner 在 voxel_grid 的范围内
    min_corner = np.maximum(min_corner, [0, 0, 0])
    max_corner = np.minimum(max_corner, voxel_grid.shape)

    # 调整 occupied_voxels 的形状以匹配 voxel_grid 的切片
    occupied_voxels = occupied_voxels[:max_corner[0]-min_corner[0], :max_corner[1]-min_corner[1], :max_corner[2]-min_corner[2]]

    voxel_grid[min_corner[0]:max_corner[0], min_corner[1]:max_corner[1], min_corner[2]:max_corner[2]] = occupied_voxels

    # 创建一个 NIfTI 图像
    nii_image = nib.Nifti1Image(voxel_grid, affine=np.eye(4))

    # 保存 NIfTI 图像
    nib.save(nii_image, nii_file)

# 使用示例
obj_to_nifti('bone_try3.obj', 'tst1_bone_2.nii', voxel_size=1.0)