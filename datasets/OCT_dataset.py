from torch.utils.data import Dataset
import os
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class OCT_dataset(Dataset):
    def __init__(self,
                 base_dir,
                 list_dir,
                 split='train',
                 use_serial=False,
                 is_train=True,
                 transform=None, ):
        self.transform = transform  # using transform in torch!
        self.split = split

        self.cirr_sample_list = open(os.path.join(list_dir, 'cirr_' + self.split + '.txt')).readlines()
        self.spec_sample_list = open(os.path.join(list_dir, 'spec_' + self.split + '.txt')).readlines()
        self.sample_list = {'cirr': self.cirr_sample_list,
                            'spec': self.spec_sample_list}

        self.cirr_size = len(self.cirr_sample_list)
        self.spec_size = len(self.spec_sample_list)

        self.data_dir = base_dir
        self.loader = pil_loader

        self.to_tensor = transforms.ToTensor()
        self.is_train = is_train
        self.use_serial = use_serial

        self.transform = transform
        self.normalize = transforms.Compose([#transforms.Resize(size=(512, 512)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

    def prepare_img(self, idx, domain):
        sample_name = self.sample_list[domain][idx].strip('\n')

        vendor = sample_name.split(' ')[0]
        slice_name = sample_name.split(' ')[1]
        slice_idx = sample_name.split(' ')[2].zfill(3)

        data_path = os.path.join(self.data_dir,
                                 vendor,
                                 slice_name,
                                 'image',
                                 slice_idx + '.png')

        image = self.loader(data_path)

        if self.transform is not None and self.is_train:
            image = self.transform(image)
        data = self.normalize(image)
        # data = F.interpolate(data.unsqueeze(1), size=[256, 256]).squeeze(1)
        return data, data_path

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """

        return max(self.cirr_size, self.spec_size)

    def __getitem__(self, idx):
        idx_cirr = idx % self.cirr_size

        if self.use_serial:   # make sure index is within then range
            idx_spec = idx % self.spec_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            idx_spec = random.randint(0, self.spec_size - 1)

        cirr_data, cirr_path = self.prepare_img(idx_cirr, 'cirr')
        spec_data, spec_path = self.prepare_img(idx_spec, 'spec')
        # images = [cirr_data, spec_data]
        # labels = torch.Tensor([1, 2])
        data = {'A': cirr_data,
                'B': spec_data,
                'A_paths': cirr_path,
                'B_paths': spec_path}

        return data

# # Test Unit
# flip = transforms.RandomHorizontalFlip(p=0.5)
#
# base_dir = '../Retouch-dataset/pre_processed/'
# list_dir = 'splits/octGAN'
#
# dataset = OCT_dataset(base_dir, list_dir, transform=flip)
# d_cirr = dataset[0]['images'][0]
# d_spec = dataset[0]['images'][1]
#
# print(dataset[0]['labels'][0], dataset[0]['labels'][1])
#
# img = d_cirr.permute(1, 2, 0).numpy()
# plt.figure()
# plt.imshow(img)
