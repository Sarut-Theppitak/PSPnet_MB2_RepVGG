# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), c='white')
        plt.imshow(image)
    plt.show()
    
# helper function for labeling multiclass masks
def labelVisualize(num_class, color_dict, img):
    img_out = img[0,:,:] if len(img.shape) == 3 else img
    img_out = np.zeros(img_out.shape + (3,))
    for i in range(num_class):
        img_out[img[i,:,:]==1, :] = COLOR_DICT[i]
    return img_out / 255
    
    
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    コマンドライン引数
    """
    parser.add_argument('-i', '--images',
                        help='image directory')
    parser.add_argument('-m', '--masks',
                        help='masks directory')
    parser.add_argument('-o', '--output',
                        help='model output directory')
    parser.add_argument('-s', '--size', type=int, default=256,
                        help='input image size (default=256)')
    parser.add_argument('-r', '--ratio', type=int, default=4,
                        help='augmentation ratio (default=4)')
    parser.add_argument('-lr', '--learning_rate', type=int,
                        default=0.0001, help='learning rate (default=0.0001)')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='training epochs (default=100)')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='training batch size (default=8)')

    FLAGS = parser.parse_args()


    # load model
    model = torch.load(FLAGS.model)
    
    # create test dataset
    x_test_dir = './data/CamVid/images/test'
    y_test_dir = './data/CamVid/masks/test'

    pipeline_aug = get_training_augmentation()
    pipeline_prepro = get_preprocessing(normalize_input)

    # class labels for cityscape dataset
    # CLASSES = ['car']
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']


    test_dataset = MyDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(),
        preprocessing=pipeline_prepro,
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)
    
    
    # test dataset without transformations for image visualization
    test_dataset_vis = MyDataset(
        x_test_dir, y_test_dir, 
        classes=CLASSES,
    )
    
    for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    gt_mask = labelVisualize(len(CLASSES), COLOR_DICT, gt_mask)
    
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = labelVisualize(len(CLASSES), COLOR_DICT, pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )