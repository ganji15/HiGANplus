import numpy as np
from skimage import metrics
from tqdm import tqdm
from lib.utils import show_image_pair


def PSNR(x_image, y_image, max_value=1.0):
    x = np.asarray(x_image, np.float32)
    y = np.asarray(y_image, np.float32)
    return metrics.peak_signal_noise_ratio(x, y, data_range=max_value)


def MSSIM(x_image, y_image, max_value=1.0):
    x = np.asarray(x_image, np.float32)
    y = np.asarray(y_image, np.float32)
    return metrics.structural_similarity(x, y,
                                         win_size=None, # default win_size is 7
                                         data_range=max_value,
                                         multichannel=(x.ndim>2))


def calculate_mssim_psnr(data_loader, generator):
    psnr, mssim = [], []
    for batch_src,  batch_gen in tqdm(zip(data_loader, generator), total=len(data_loader)):
        src_imgs, src_img_lens = batch_src['org_imgs'], batch_src['org_img_lens']
        gen_imgs, gen_img_lens = batch_gen['org_imgs'], batch_gen['org_img_lens']
        for src_img, src_img_len, gen_img, gen_img_len in \
                zip(src_imgs, src_img_lens, gen_imgs, gen_img_lens):
            # assert gen_img_len == src_img_len, "gen_img_len %d != src_img_len %d"%(gen_img_len, src_img_len)
            # show_image_pair(src_img[0, :, :src_img_len].cpu().numpy(),
            #                 gen_img[0, :, :gen_img_len].cpu().numpy(),
            #                 'src_img', 'gen_img')
            if gen_img_len != src_img_len:
                print("gen_img_len %d != src_img_len %d"%(gen_img_len, src_img_len))
                show_image_pair(src_img[0, :, :src_img_len].cpu().numpy(),
                                gen_img[0, :, :gen_img_len].cpu().numpy(),
                                'src_img', 'gen_img')
            src_img = (src_img[:, :, :src_img_len].permute(1, 2, 0).cpu().numpy() + 1) / 2
            gen_img = (gen_img[:, :, :gen_img_len].permute(1, 2, 0).cpu().numpy() + 1) / 2
            psnr.append(PSNR(src_img, gen_img, max_value=1.0))
            mssim.append(MSSIM(src_img, gen_img, max_value=1.0))

    count = len(psnr)
    psnr = sum(psnr) / count
    mssim = sum(mssim) / count
    return {'psnr': psnr, 'mssim': mssim}