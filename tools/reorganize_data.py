import os
import shutil


def reorganize_data_folders(root_path='./data', output_path='./data/all'):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(os.path.join(output_path, 'images'))
    os.makedirs(os.path.join(output_path, 'labels'))
    os.makedirs(os.path.join(output_path, '2nd grader'))

    # FDA
    for sub_dir in os.listdir(os.path.join(root_path, 'FDA', 'Images and GT')):
        if os.path.isdir(os.path.join(root_path, 'FDA', 'Images and GT', sub_dir)):
            for img_filename in os.listdir(os.path.join(root_path, 'FDA', 'Images and GT', sub_dir)):
                if img_filename.endswith('.tif'):
                    name_parts = img_filename.split('_')
                    subject_num = name_parts[1]
                    retina_location = name_parts[-1][:-4]
                    target_filename = '_'.join(['FDA', subject_num, retina_location, sub_dir])

                    shutil.copy(os.path.join(root_path, 'FDA', 'Images and GT', sub_dir, img_filename),
                                os.path.join(output_path, 'images', target_filename + '.tif'))
                    shutil.copy(os.path.join(root_path, 'FDA', 'Images and GT', sub_dir, img_filename[:-4] + '_marker.mat'),
                                os.path.join(output_path, 'labels', target_filename + '_marker.mat'))
                    shutil.copy(os.path.join(root_path, 'FDA', '2nd grader', sub_dir, img_filename[:-4] + '_marker.mat'),
                                os.path.join(output_path, '2nd grader', target_filename + '_marker.mat'))

    # IU
    for img_filename in os.listdir(os.path.join(root_path, 'IU', 'Images and GT')):
        if img_filename.endswith('.tif'):
            name_parts = img_filename.split('_')
            subject_num = name_parts[0]
            retina_location = name_parts[1]
            target_filename = '_'.join(['IU', subject_num, retina_location])

            shutil.copy(os.path.join(root_path, 'IU', 'Images and GT', img_filename),
                        os.path.join(output_path, 'images', target_filename + '.tif'))
            shutil.copy(os.path.join(root_path, 'IU', 'Images and GT', img_filename[:-4] + '_marker.mat'),
                        os.path.join(output_path, 'labels', target_filename + '_marker.mat'))
            shutil.copy(os.path.join(root_path, 'IU', '2nd grader', img_filename[:-4] + '_marker.mat'),
                        os.path.join(output_path, '2nd grader', target_filename + '_marker.mat'))


if __name__ == '__main__':
    reorganize_data_folders()
