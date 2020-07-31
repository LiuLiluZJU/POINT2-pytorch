import numpy as np
import scipy.ndimage as ndimage
import os
import matplotlib.pyplot as plt
import h5py
import random
import SimpleITK as sitk
from SiddonGpuPy import pySiddonGpu
from Transformation3DGpuPy import pyTransformation3DGpu


def get_rotation_mat_single_axis( axis, angle ):

    """It computes the 3X3 rotation matrix relative to a single rotation of angle(rad) 
    about the axis(string 'x', 'y', 'z') for a righr handed CS"""

    if axis == 'x' : return np.array(([1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]))

    if axis == 'y' : return np.array(([np.cos(angle),0,np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]))

    if axis == 'z' : return np.array(([np.cos(angle),-np.sin(angle),0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]))


def get_rigid_motion_mat_from_euler( alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z ):
    
    """It computes the 4X4 rigid motion matrix given a sequence of 3 Euler angles about the 3 axes 1,2,3 
    and the translation vector t_x, t_y, t_z"""

    rot1 = get_rotation_mat_single_axis( axis_1, alpha )
    rot2 = get_rotation_mat_single_axis( axis_2, beta )
    rot3 = get_rotation_mat_single_axis( axis_3, gamma )

    rot_mat = np.dot(rot1, np.dot(rot2,rot3))

    t = np.array(([t_x], [t_y], [t_z]))

    output = np.concatenate((rot_mat, t), axis = 1)

    return np.concatenate((output, np.array([[0.,0.,0.,1.]])), axis = 0)


def transformImage(image, matrix):

    spacing = np.asarray(image.GetSpacing())
    size = np.asarray(image.GetSize())
    origin = np.asarray(image.GetOrigin())
    center = np.asarray(origin) + np.multiply(spacing, np.divide(size, 2.)) - np.divide(spacing, 2.)

    image_array = sitk.GetArrayFromImage(image)
    image_array_1d = np.ravel(image_array, order='C')

    # GPU实现CT旋转平移
    numThreadsPerBlock = np.array([8, 8, 8])
    transformer = pyTransformation3DGpu(numThreadsPerBlock.astype(np.int32),
                                            image_array_1d.astype(np.float32),
                                            origin.astype(np.float32),
                                            size.astype(np.int32),
                                            spacing.astype(np.float32),
                                            center.astype(np.float32))
    matrix_1d = np.ravel(matrix, order='C')
    output_image_array_1d = transformer.Transform(matrix_1d.astype(np.float32))
    output_image_array = np.reshape(output_image_array_1d, (size[2], size[1], size[0]), order='C')

    # 释放显存
    transformer.delete()

    return output_image_array



def projection2D(image, fiducial_points, source, center, DRRphy_array, DRRspacing, DRRsize, transform):

    """返回三维参考点的二维投影(x, y, z)：
    x-图像坐标第二维；y-图像坐标第一维；z-点的序号"""

    spacing = np.asarray(image.GetSpacing())
    size = np.asarray(image.GetSize())
    origin = np.asarray(image.GetOrigin())
    center = np.asarray(origin) + np.multiply(spacing, np.divide(size, 2.)) - np.divide(spacing, 2.)

    fiducial_points_phy = \
        np.multiply(fiducial_points, np.tile(spacing, (fiducial_points.shape[0], 1))) + \
        np.tile(origin, (fiducial_points.shape[0], 1))

    # Move source point with transformation matrix, transform around volume center (subtract volume center point)
    source_transformed = np.dot(transform, np.array(
        [source[0] - center[0], source[1] - center[1], source[2] - center[2], 1.]).T)[
                            0:3]
    source = np.array([source_transformed[0] + center[0], source_transformed[1] + center[1],
                                source_transformed[2] + center[2]])

    # Get array of physical coordinates of the transformed DRR
    Tn = np.array([[1., 0., 0., center[0]],
                       [0., 1., 0., center[1]],
                       [0., 0., 1., center[2]],
                       [0., 0., 0., 1.]])
    invTn = np.linalg.inv(Tn)
    DRRphy_array_reshaped = DRRphy_array.reshape((DRRsize[0] * DRRsize[1], 3), order='C')
    DRRphy_array_augmented =  np.dot(invTn, np.concatenate((DRRphy_array_reshaped, np.ones((len(DRRphy_array_reshaped), 1))), axis = 1).T)
    DRRphy_array_augmented_transformed = np.dot(transform, DRRphy_array_augmented)
    DRRphy_array_transformed = np.transpose(np.dot(Tn, DRRphy_array_augmented_transformed)[0:3], (1, 0))
    DRRphy_array_transf_to_ravel = DRRphy_array_transformed.reshape(
        (DRRsize[0], DRRsize[1], 3), order='C')
    DRRorigin = DRRphy_array_transf_to_ravel[0][0]
    # DRRcenter = DRRphy_array_transf_to_ravel[int(DRRsize[0] / 2)][int(DRRsize[1]/ 2)]

    DRRorigin_to_transform = DRRphy_array_reshaped[0]
    DRRcenter_to_transform = DRRorigin_to_transform + np.multiply(DRRspacing, np.divide(DRRsize, 2.)) - np.divide(DRRspacing, 2.)
    DRRcenter_transformed = np.dot(transform, np.array(
        [DRRcenter_to_transform[0] - center[0], DRRcenter_to_transform[1] - center[1], DRRcenter_to_transform[2] - center[2], 1.]).T)[0:3]
    DRRcenter = np.array([DRRcenter_transformed[0] + center[0], DRRcenter_transformed[1] + center[1], DRRcenter_transformed[2] + center[2]])

    vector_DRR_x = np.dot(transform[0:3, 0:3], [1, 0, 0])
    vector_DRR_y = np.dot(transform[0:3, 0:3], [0, 1, 0])

    DRR_dest_index_array = []
    for i in range(fiducial_points_phy.shape[0]):
        vector_verticle = DRRcenter - source
        vector_fiducial = fiducial_points_phy[i] - source
        factor = np.dot(vector_fiducial, vector_verticle / np.linalg.norm(vector_verticle)) / np.linalg.norm(vector_verticle)
        DRR_dest = vector_fiducial / factor + source
        vector_DRR_dest = DRR_dest - DRRorigin
        DRR_dest_index_x = np.round(np.dot(vector_DRR_dest, vector_DRR_x) / DRRspacing[0])
        DRR_dest_index_y = np.round(np.dot(vector_DRR_dest, vector_DRR_y) / DRRspacing[1])
        # DRR_dest_index_z = i  # indexing point's identity
        if DRR_dest_index_x < 0 or DRR_dest_index_x >= DRRsize[1] or \
            DRR_dest_index_y < 0 or DRR_dest_index_y >= DRRsize[0]:
            continue
        DRR_dest_index_array.append([DRR_dest_index_x, DRR_dest_index_y])
    DRR_dest_index_array = np.array(DRR_dest_index_array, dtype=np.int)
    # print(DRR_dest_index_array)

    return DRR_dest_index_array


projector_info = {'Name': 'SiddonGpu',
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1,
                  'DRRsize_x': 200,
                  'DRRsize_y': 200,
                  'focal_lenght': 1800,
                  'DRR_ppx': 0,
                  'DRR_ppy': 0,
                  'DRRspacing_x': 1,
                  'DRRspacing_y': 1}

data_path_list = os.listdir('/home/leko/POINT2-data/data_h5_cq500')
data_count = 1

for data_path in data_path_list:
    # Read CT
    # data_path = 'LIDC-IDRI-0001.20000101.3000566.1'
    file_path = os.path.join('/home/leko/POINT2-data/data_h5_cq500', data_path, 'ct_xray_data.h5')
    h5_file = h5py.File(file_path, 'r')
    input_ct_array = h5_file['ct'][()].astype(np.float64)
    input_ct_array = input_ct_array[::-1, :, :]

    # 读取CT图像信息
    movSpacing = np.array([1, 1, 1])
    movSize = np.array(input_ct_array.shape)[::-1]
    # movSpacing = np.asarray(input_ct_image.GetSpacing())
    # movSize = np.asarray(input_ct_image.GetSize())
    movOrigin = np.array([0, 0, 0])
    movCenter = np.asarray(movOrigin) + np.multiply(movSpacing, np.divide(movSize, 2.)) - np.divide(movSpacing, 2.)

    # 计算边界平面
    X0 = movCenter[0] - movSpacing[0] * movSize[0] * 0.5
    Y0 = movCenter[1] - movSpacing[1] * movSize[1] * 0.5
    Z0 = movCenter[2] - movSpacing[2] * movSize[2] * 0.5

    # 设置位姿扰动值
    # d_alpha = random.uniform(-10, 10)
    # d_beta = random.uniform(-10, 10)
    # d_gamma = random.uniform(-10, 10)
    # d_x = random.uniform(-20, 20)
    # d_y = random.uniform(-20, 20)
    # d_z = random.uniform(-20, 20)
    d_alpha = 0
    d_beta = 0
    d_gamma = 0
    d_x = 0
    d_y = 0
    d_z = 0
    d_params = np.array([d_alpha, d_beta, d_gamma, d_x, d_y, d_z])
    Tr_delta = get_rigid_motion_mat_from_euler(np.deg2rad(d_alpha), 'x', np.deg2rad(d_beta), 'y', np.deg2rad(d_gamma), 'z', d_x, d_y, d_z)

    ######################################
    #------------从CT产生DRR---------------
    ######################################

    # 将numpy数组转换为simpleitk图像
    new_input_ct_image = sitk.GetImageFromArray(input_ct_array)
    new_input_ct_image.SetOrigin(movOrigin.astype(np.float64))
    new_input_ct_image.SetSpacing(movSpacing.astype(np.float64))

    # 对CT进行旋转、平移（用于产生DRR）
    new_input_ct_array = transformImage(new_input_ct_image, Tr_delta)
    # plt.subplot(131)
    # plt.imshow(np.squeeze(np.mean(new_input_ct_array, axis=0)), cmap='gray')
    # plt.subplot(132)
    # plt.imshow(np.squeeze(np.mean(new_input_ct_array, axis=1)), cmap='gray')
    # plt.subplot(133)
    # plt.imshow(np.squeeze(np.mean(new_input_ct_array, axis=2)), cmap='gray')
    # plt.show()

    # 设置GPU参数
    NumThreadsPerBlock = np.array([projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'],
                            projector_info['threadsPerBlock_z']]).astype(np.int32)
    DRRsize_forGpu = np.array([projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1]).astype(np.int32)
    MovSize_forGpu = np.array([movSize[0], movSize[1], movSize[2]]).astype(np.int32)
    MovSpacing_forGpu = np.array([movSpacing[0], movSpacing[1], movSpacing[2]]).astype(np.float32)
    MovCenter_forGpu = movCenter.astype(np.float32)
    movImgArray_1d = np.ravel(new_input_ct_array.copy(), order='C').astype(np.float32)

    # 定义光源位置
    source = np.zeros(3, dtype=np.float32)
    source[0] = movCenter[0]
    source[1] = movCenter[1]
    source[2] = movCenter[2] - projector_info['focal_lenght'] / 2.

    # 定义DRR参数
    DRRsize = [0] * 3
    DRRsize[0] = projector_info['DRRsize_x']
    DRRsize[1] = projector_info['DRRsize_y']
    DRRsize[2] = 1

    DRRspacing = [0] * 3
    DRRspacing[0] = projector_info['DRRspacing_x']
    DRRspacing[1] = projector_info['DRRspacing_y']
    DRRspacing[2] = 1

    DRRorigin = [0] * 3
    DRRorigin[0] = movCenter[0] - projector_info['DRR_ppx'] - DRRspacing[0] * (DRRsize[0] - 1.) / 2.
    DRRorigin[1] = movCenter[1] - projector_info['DRR_ppy'] - DRRspacing[1] * (DRRsize[1] - 1.) / 2.
    DRRorigin[2] = movCenter[2] + projector_info['focal_lenght'] / 2.
    
    DRR = sitk.Image([DRRsize[0], DRRsize[1], 1], sitk.sitkFloat64)
    DRR.SetOrigin(DRRorigin)
    DRR.SetSpacing(DRRspacing)
    PhysicalPointImagefilter = sitk.PhysicalPointImageSource()
    PhysicalPointImagefilter.SetReferenceImage(DRR)
    sourceDRR = PhysicalPointImagefilter.Execute()
    sourceDRR_array_to_reshape = sitk.GetArrayFromImage(sourceDRR)
    sourceDRR_array_1d = np.ravel(sourceDRR_array_to_reshape, order='C').astype(np.float32)

    # 定义虚假的固定X光
    fixedImgArray_1d = np.zeros(DRRsize[0] * DRRsize[1], dtype=np.float32)

    # 初始化投影器
    projector = pySiddonGpu(NumThreadsPerBlock,
                            movImgArray_1d,
                            MovSize_forGpu,
                            MovSpacing_forGpu,
                            X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                            DRRsize_forGpu,
                            fixedImgArray_1d,
                            sourceDRR_array_1d,
                            source.astype(np.float32),
                            MovCenter_forGpu)

    # 设置变换参数
    Tr_ap = get_rigid_motion_mat_from_euler(np.deg2rad(-90), 'x', np.deg2rad(0), 'y', np.deg2rad(0), 'z', 0, -700, 0)  # 注意：此旋转方式为“XYZ”
    Tr_lat = get_rigid_motion_mat_from_euler(np.deg2rad(0), 'x', np.deg2rad(90), 'y', np.deg2rad(-90), 'z', -700, 0, 0)

    # 产生DRR
    invT_ap_1d = np.ravel(Tr_ap, order='C').astype(np.float32)
    drr1_to_reshape = projector.generateDRR(invT_ap_1d)
    drr1 = np.reshape(drr1_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    projector.computeMetric()
    invT_lat_1d = np.ravel(Tr_lat, order='C').astype(np.float32)
    drr2_to_reshape = projector.generateDRR(invT_lat_1d)
    drr2 = np.reshape(drr2_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    projector.computeMetric()

    # DRR像素规范化
    drr1 = (drr1 - np.min(drr1)) / (np.max(drr1) - np.min(drr1)) * 255
    drr2 = (drr2 - np.min(drr2)) / (np.max(drr2) - np.min(drr2)) * 255
    
    # 显示DRR
    # plt.subplot(121)
    # plt.imshow(drr1, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(drr2, cmap='gray')
    # plt.show()

    # 释放显存
    projector.delete()

    # 随机生成3D参考点
    correspondence_2D_ap = []
    correspondence_2D_lat = []
    fiducial_3D = []
    point_count = 0
    while(point_count < 20):
    
        fiducial_point = np.array([np.random.randint(0, 255, size=3)])

        # 参考点投影到2D平面
        annotation_points_drr_ap = projection2D(new_input_ct_image, fiducial_point, source, movCenter, sourceDRR_array_to_reshape, DRRspacing, DRRsize, Tr_ap)
        annotation_points_xray_ap = projection2D(new_input_ct_image, fiducial_point, source, movCenter, sourceDRR_array_to_reshape, DRRspacing, DRRsize, Tr_ap)
        annotation_points_drr_lat = projection2D(new_input_ct_image, fiducial_point, source, movCenter, sourceDRR_array_to_reshape, DRRspacing, DRRsize, Tr_lat)
        annotation_points_xray_lat = projection2D(new_input_ct_image, fiducial_point, source, movCenter, sourceDRR_array_to_reshape, DRRspacing, DRRsize, Tr_lat)

        print(annotation_points_drr_ap)
        print(annotation_points_drr_lat)
        if (annotation_points_drr_ap.any() and annotation_points_xray_ap.any() and
                annotation_points_drr_lat.any() and annotation_points_xray_lat.any()):
            annotation_points_zipped_ap = np.hstack([annotation_points_drr_ap, annotation_points_xray_ap])
            annotation_points_zipped_lat = np.hstack([annotation_points_drr_lat, annotation_points_xray_lat])
            correspondence_2D_ap.append(annotation_points_zipped_ap)
            correspondence_2D_lat.append(annotation_points_zipped_lat)
            fiducial_3D.append(fiducial_point)
            point_count += 1

            # 简洁投影公式
            # fiducial_point_phy = \
            #         np.multiply(fiducial_point, np.tile(movSpacing, (fiducial_point.shape[0], 1))) + \
            #         np.tile(movOrigin, (fiducial_point.shape[0], 1))
            # X = fiducial_point - movCenter
            # d = 1800
            # c = 900  # 注意：c表示成像系统在Rt变换前的中心（初始化为焦距的一半）
            # K = np.array([[d, 0, 0],
            #                 [0, d, 0],
            #                 [0, 0, 1]])
            # h = np.array([[0, 0, c]]).T
            # Tr_ap_inv = np.linalg.inv(Tr_ap)
            # R_view1 = Tr_ap_inv[0 : 3, 0 : 3]
            # t_view1 = Tr_ap_inv[:3, 3].T
            # x_dot1 = np.dot(K, np.dot(np.hstack([R_view1, np.array([t_view1]).T + h]), np.append(X, 1).T))
            # x_dot1 = (x_dot1 / x_dot1[-1])[:2]
            # point_dot1 = x_dot1[:2] + np.array([127.5, 127.5])
            # print(point_dot1)
            # Tr_lat_inv = np.linalg.inv(Tr_lat)
            # R_view2 = Tr_lat_inv[0 : 3, 0 : 3]
            # t_view2 = Tr_lat_inv[:3, 3].T
            # x_dot2 = np.dot(K, np.dot(np.hstack([R_view2, np.array([t_view2]).T + h]), np.append(X, 1).T))
            # x_dot2 = (x_dot2 / x_dot2[-1])[:2]
            # point_dot2 = x_dot2[:2] + np.array([127.5, 127.5])
            # print(point_dot2)

            # D_x1 = np.hstack([np.array([[-d, 0], [0, -d]]), np.array([x_dot1]).T])
            # D_x2 = np.hstack([np.array([[-d, 0], [0, -d]]), np.array([x_dot2]).T])

            # A = np.squeeze(np.hstack([[np.dot(D_x1, R_view1)], [np.dot(D_x2, R_view2)]]))
            # b = np.hstack([-c * x_dot1 - np.dot(D_x1, t_view1), -c * x_dot2 - np.dot(D_x2, t_view2)])

            # X_3d = np.dot(np.linalg.pinv(A), b)
            # print(X_3d, X)

    correspondence_2D_ap = np.squeeze(np.asarray(correspondence_2D_ap))
    correspondence_2D_lat = np.squeeze(np.asarray(correspondence_2D_lat))
    fiducial_3D = np.squeeze(np.asarray(fiducial_3D))
    print(correspondence_2D_ap)
    print(correspondence_2D_lat)
    print(fiducial_3D)

    # plt.subplot(121)
    # plt.imshow(drr1, cmap='gray')
    # plt.scatter(correspondence_2D_ap[:, 0], correspondence_2D_ap[:, 1], marker='+')
    # plt.subplot(122)
    # plt.imshow(drr2, cmap='gray')
    # plt.scatter(correspondence_2D_lat[:, 2], correspondence_2D_lat[:, 3], marker='x')
    # plt.show()

    h5_file = h5py.File("/home/leko/POINT2-data/data_multiview_cq500/" + str(data_count) + ".h5", 'w')
    h5_file.create_dataset('input_drr_ap', data=drr1)
    h5_file.create_dataset('input_xray_ap', data=drr1)
    h5_file.create_dataset('input_drr_lat', data=drr2)
    h5_file.create_dataset('input_xray_lat', data=drr2)
    h5_file.create_dataset('correspondence_2D_ap', data=correspondence_2D_ap)
    h5_file.create_dataset('correspondence_2D_lat', data=correspondence_2D_lat)
    h5_file.create_dataset('fiducial_3D', data=fiducial_3D)
    h5_file.close()
    data_count += 1