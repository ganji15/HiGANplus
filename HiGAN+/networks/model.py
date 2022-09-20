import torch, os
from PIL import Image
from munch import Munch
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from distance import levenshtein
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from metric.fid_kid_is import calculate_fid_kid_is
from metric.mssim_psnr import calculate_mssim_psnr
from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, rescale_images, rescale_images2, \
                            words_to_images, ctc_greedy_decoder, extract_all_patches
from networks.BigGAN_networks import Generator, Discriminator, PatchDiscriminator
from networks.module import Recognizer, WriterIdentifier, StyleEncoder, StyleBackbone
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
from lib.alphabet import strLabelConverter, get_lexicon, get_true_alphabet, Alphabets
from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string, AverageMeter, plot_heatmap
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.loss import recn_l1_loss, CXLoss, KLloss


class BaseModel(object):
    def __init__(self, opt, log_root='./'):
        self.opt = opt
        self.local_rank = opt.local_rank if 'local_rank' in opt else -1
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.optimizers = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None
        alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)

    def print(self, info):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger or self.writer:
            return

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.writer = SummaryWriter(log_dir=self.log_root)

        opt_str = option_to_string(self.opt)
        with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
            f.writelines(opt_str)
        print('log_root: ', self.log_root)
        self.logger = get_logger(self.log_root)

    def info(self, extra=None):
        self.print("RUNDIR: {}".format(self.log_root))
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print('=' * 20)

    def save(self, tag='best', epoch_done=0, **kwargs):
        ckpt = {}
        for model in self.models.values():
            ckpt[type(model).__name__] = model.state_dict()

        for key, optim in self.optimizers.items():
            ckpt['OPT.' + key] = optim.state_dict()

        for key, val in kwargs.items():
            ckpt[key] = val

        ckpt['Epoch'] = epoch_done
        ckpt_save_path = os.path.join(self.log_root, self.opt.training.ckpt_dir, tag + '.pth')
        torch.save(ckpt, ckpt_save_path)

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print('load checkpoint from ', ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location)

        if ckpt is None:
            return

        models = self.models.values() if len(modules) == 0 else modules
        for model in models:
            try:
                model.load_state_dict(ckpt.pop(type(model).__name__))
            except Exception as e:
                print('Load %s failed'%type(model).__name__)

        for key in self.optimizers.keys():
            try:
                self.optimizers[key].load_state_dict(ckpt.pop('OPT.' + key))
            except Exception as e:
                print('Load %s failed'%('OPT.' + key))

        ckpt['Epoch'] = 0 if 'Epoch' not in ckpt else ckpt['Epoch']
        return ckpt['Epoch']

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()

    def validate(self, *args, **kwargs):
        yield NotImplementedError()


    def train(self):
        yield NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(AdversarialModel, self).__init__(opt, log_root)

        self.lexicon = get_lexicon(self.opt.training.lexicon,
                                   get_true_alphabet(opt.dataset),
                                   max_length=self.opt.training.max_word_len)
        self.max_valid_image_width = self.opt.char_width * self.opt.training.max_word_len
        self.vae_mode = self.opt.training.vae_mode
        self.collect_fn = get_collect_fn(self.opt.training.sort_input, sort_style=True)
        self.train_loader = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split,
                        recogn_aug=True, wid_aug=True, process_style=True),
            batch_size=opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4,
            drop_last=True
        )

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split,
                        recogn_aug=False, wid_aug=False, process_style=True),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        self.tst_loader2 = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split,
                        recogn_aug=False, wid_aug=False, process_style=True),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        self.models = None

    def train(self):
        raise NotImplementedError()

    def sample_images(self, iteration_done=0):
        self.set_mode('eval')

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)

        real_imgs, real_img_lens = batch['style_imgs'].to(device), batch['style_img_lens'].to(device)
        real_lbs, real_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)

        with torch.no_grad():
            self.eval_z.sample_()
            recn_imgs = None
            if 'E' in self.models:
                enc_z = self.models.E(real_imgs, real_img_lens, self.models.B)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(self.eval_z, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(self.eval_y, self.lexicon,
                                         self.opt.training.capitalize_ratio,
                                         self.opt.training.blank_ratio)
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
            fake_imgs = self.models.G(self.eval_z, fake_lbs, fake_lb_lens)
            style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)

            max_img_len = max([real_imgs.size(-1), fake_real_imgs.size(-1), fake_imgs.size(-1)])
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            real_imgs = F.pad(real_imgs, [0, max_img_len - real_imgs.size(-1), 0, 0], value=-1.)
            fake_real_imgs = F.pad(fake_real_imgs, [0, max_img_len - fake_real_imgs.size(-1), 0, 0], value=-1.)
            fake_imgs = F.pad(fake_imgs, [0, max_img_len - fake_imgs.size(-1), 0, 0], value=-1.)
            recn_imgs = F.pad(recn_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=-1.) \
                        if recn_imgs is not None else None
            style_imgs = F.pad(style_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=-1.)

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [real_labels.cpu(), real_imgs.cpu(), fake_real_imgs.cpu(),
                                   fake_imgs.cpu(), style_imgs.cpu(), rand_labels.cpu()]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(1, 3, 1, 1)
                res_img = draw_image(1 - sample_imgs.data, nrow=self.opt.training.sample_nrow, normalize=True)
                save_path = os.path.join(self.log_root, self.opt.training.sample_dir,
                                         'iter_{}.png'.format(iteration_done))
                im = Image.fromarray(res_img)
                im.save(save_path)
                if self.writer:
                    self.writer.add_image('Image', res_img.transpose((2, 0, 1)), iteration_done)
            except RuntimeError as e:
                print(e)

    def image_generator(self, style_dloader, use_rand_corpus=False, style_guided=True, n_repeats=1):
        device = self.device
        word_idx_sampler = None
        if use_rand_corpus:
            word_idx_sampler = prepare_y_dist(style_dloader.batch_size,
                                              len(self.lexicon),
                                              self.device,
                                              seed=self.opt.seed)

        if style_guided and not use_rand_corpus:
            n_repeats = 1

        with torch.no_grad():
            for _ in range(n_repeats):
                for batch in style_dloader:
                    fake_batch = {}
                    style_imgs, style_img_lens = batch['style_imgs'].to(device), batch['style_img_lens'].to(device)
                    style_lbs, style_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)
                    if use_rand_corpus:
                        word_idx_sampler.sample_()
                        sampled_words = idx_to_words(word_idx_sampler[:style_imgs.size(0)],
                                                     self.lexicon, self.opt.training.capitalize_ratio,
                                                     blank_ratio=0)
                        content_lbs, content_lb_lens = self.label_converter.encode(sampled_words)
                    else:
                        content_lbs, content_lb_lens = style_lbs, style_lb_lens

                    fake_batch['lbs'], fake_batch['lb_lens'] = content_lbs.to(device), content_lb_lens.to(device)

                    if style_guided:
                        enc_z = self.models.E(style_imgs.to(device), style_img_lens.to(device), self.models.B)
                    else:
                        enc_z = torch.randn(style_lb_lens.size(0), self.models.G.style_dim).to(device)

                    fake_batch['style_imgs'] = self.models.G(enc_z, content_lbs, content_lb_lens)
                    fake_batch['style_img_lens'] = fake_batch['lb_lens'] * self.opt.char_width
                    fake_batch['wids'] = batch['wids']

                    fake_batch['org_imgs'], fake_batch['org_img_lens'] =\
                                        rescale_images(fake_batch['style_imgs'],
                                        fake_batch['style_img_lens'],
                                        batch['org_img_lens'])

                    yield fake_batch

    def validate(self, style_guided=True, test_stage=False, *args, **kwargs):
        self.set_mode('eval')
        # style images are resized
        eval_dloader = DataLoader(
            get_dataset(self.opt.valid.dset_name, self.opt.valid.dset_split, process_style=True),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )

        if 'E' not in self.models:
            style_guided = False
            n_rand_repeat = 1
        else:
            n_rand_repeat = 1 if style_guided and not self.opt.valid.use_rand_corpus \
                              else self.opt.valid.n_rand_repeat

        def get_generator():
            generator = self.image_generator(eval_dloader, self.opt.valid.use_rand_corpus,
                                             style_guided, n_rand_repeat)
            return generator

        if test_stage:
            res = calculate_fid_kid_is(self.opt.valid, eval_dloader, get_generator(), n_rand_repeat, self.device)
        else:
            res = calculate_fid_kid_is(self.opt.valid, eval_dloader, get_generator(), n_rand_repeat, self.device, crop=True)

        if test_stage:
            if not self.opt.valid.use_rand_corpus:
                psnr_mssim = calculate_mssim_psnr(eval_dloader, get_generator())
                res['psnr'] = psnr_mssim['psnr']
                res['mssim'] = psnr_mssim['mssim']
            res['cer'], res['wer'] = self.validate_ocr(get_generator(), n_iters=len(eval_dloader) * n_rand_repeat)
            if style_guided:
                wier = self.validate_wid(get_generator(), real_dloader=eval_dloader, split=self.opt.valid.dset_split)
                res['wier'] = wier

        return res

    def validate_ocr(self, dloader, n_iters):
        self.set_mode('eval')
        recognizer = Recognizer(**self.opt.OcrModel).to(self.device)
        r_dict = torch.load(self.opt.training.pretrained_r)['Recognizer']
        recognizer.load_state_dict(r_dict, self.device)
        recognizer.eval()
        print('load pretrained recognizer: ', self.opt.training.pretrained_r)
        ctc_len_scale = self.models.R.len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dloader), total=n_iters):
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device), batch['style_img_lens'].to(self.device)
                logits = recognizer(real_imgs, real_img_lens)
                logits = torch.nn.functional.softmax(logits, dim=2).detach()

                logits = logits.cpu().numpy()
                word_preds = []
                for logit, img_len in zip(logits, batch['style_img_lens'].cpu().numpy()):
                    label = ctc_greedy_decoder(logit[:img_len // ctc_len_scale])
                    word_preds.append(self.label_converter.decode(label))
                word_reals = self.label_converter.decode(batch['lbs'], batch['lb_lens'])
                for word_pred, word_real in zip(word_preds, word_reals):
                    char_tran = levenshtein(word_pred, word_real)
                    char_trans += char_tran
                    total_chars += len(word_real)
                    total_words += 1
                    if char_tran > 0:
                        word_trans += 1

        for model in self.models.values():
            model.train()
        cer = char_trans * 1.0 / total_chars
        wer = word_trans * 1.0 / total_words
        print('CER:{:.4f}  WER:{:.4f}'.format(cer, wer))
        return cer, wer

    def validate_wid(self, generator, real_dloader, split='test'):
        if split == 'test':
            assert os.path.exists(self.opt.valid.pretrained_test_w)
            w_dict = torch.load(self.opt.valid.pretrained_test_w, self.device)
            test_writer = WriterIdentifier(**self.opt.valid.test_wid_model).to(self.device)
            test_writer.load_state_dict(w_dict['WriterIdentifier'])
            test_writer_backbone = StyleBackbone(**self.opt.StyBackbone).to(self.device)
            test_writer_backbone.load_state_dict(w_dict['StyleBackbone'])
            writer_identifier = test_writer
            writer_backbone = test_writer_backbone
            print('load pretrained test_writer_identifier: ', self.opt.valid.pretrained_test_w)
        else:
            writer_identifier = WriterIdentifier(**self.opt.WidModel).to(self.device)
            writer_backbone = StyleBackbone(**self.opt.StyBackbone).to(self.device)
            print('load pretrained writer_identifier: ', self.opt.training.pretrained_w)
            w_dict = torch.load(self.opt.training.pretrained_w, self.device)
            writer_identifier.load_state_dict(w_dict['WriterIdentifier'])
            writer_backbone.load_state_dict(w_dict['StyleBackbone'])

        writer_identifier.eval(), writer_backbone.eval()
        with torch.no_grad():
            n_iters = len(real_dloader)

            acc_counts = 0.
            total_counts = 0.
            for i, (batch_real, batch_fake) \
                in tqdm(enumerate(zip(real_dloader, generator)), total=n_iters):
                # predicting pesudo labels
                real_wid_logits = writer_identifier(batch_real['style_imgs'].to(self.device),
                                                batch_real['style_img_lens'].to(self.device),
                                                writer_backbone)
                _, real_preds = torch.max(real_wid_logits.data, dim=1)

                # predicting pesudo labels
                fake_wid_logits = writer_identifier(batch_fake['style_imgs'].to(self.device),
                                                batch_fake['style_img_lens'].to(self.device),
                                                writer_backbone)
                _, fake_preds = torch.max(fake_wid_logits.data, dim=1)
                acc_counts += real_preds.eq(fake_preds.to(self.device)).sum().item()
                total_counts += real_preds.size(0)

            wier = 1 - acc_counts * 1. / total_counts

        for model in self.models.values():
            model.train()
        print('WID_wier:{:.2f}'.format(wier))
        return wier

    def eval_interp(self):
        self.set_mode('eval')

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                # style0 = torch.zeros((1, self.opt.GenModel.style_dim)) + 1e-1
                # style1 = torch.ones_like(style0) - 1e-1
                style0 = torch.randn((1, self.opt.EncModel.style_dim))
                style1 = torch.randn(style0.size())

                styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                styles = torch.cat(styles, dim=0).float().to(self.device)

                fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                         fake_lb_lens.repeat(nrow * ncol).to(self.device)
                gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow * ncol):
                    plt.subplot(nrow, ncol, i + 1)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_style(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset(self.opt.dataset, self.opt.training.dset_split, process_style=True),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                batch = next(iter(tst_loader))
                imgs, img_lens, lbs, lb_lens = \
                    batch['style_imgs'], batch['style_img_lens'], batch['lbs'], batch['lb_lens']
                real_imgs, real_img_lens = imgs.to(self.device), img_lens.to(self.device)
                if len(texts) == 1:
                    fake_lbs = self.label_converter.encode(texts)
                    fake_lbs = torch.LongTensor(fake_lbs)
                    fake_lb_lens = torch.IntTensor([len(texts[0])])
                else:
                    fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.B).unsqueeze(1).\
                                repeat(1, ncol, 1).view(nrow * ncol, self.opt.EncModel.style_dim)

                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                recn_styles = self.models.E(gen_imgs, fake_lb_lens * self.opt.char_width, self.models.B)
                gen_imgs, gen_img_lens = rescale_images2(gen_imgs, fake_lb_lens * self.opt.char_width, fake_lb_lens,
                                           batch['org_img_lens'].repeat_interleave(ncol).to(self.device),
                                           batch['lb_lens'].repeat_interleave(ncol).to(self.device))
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                real_imgs = torch.nn.functional.pad(batch['org_imgs'],
                                                    [0, gen_imgs.shape[-1] - batch['org_imgs'].size(-1), 0, 0],
                                                    mode='constant', value=-1)
                real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 1)
                    # plt.imshow(real_imgs[i, :, :real_img_lens[i]], cmap='gray')
                    plt.imshow(real_imgs[i], cmap='gray')
                    plt.axis('off')
                    for j in range(ncol):
                        plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 2 + j)
                        # plt.imshow(gen_imgs[i * ncol + j, :, :gen_img_lens[i * ncol + j]], cmap='gray')
                        plt.imshow(gen_imgs[i * ncol + j], cmap='gray')
                        plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_rand(self):
        self.set_mode('eval')

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            rand_z = prepare_z_dist(nrow, self.opt.EncModel.style_dim, self.device)
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                if len(texts) == 1:
                    fake_lbs = self.label_converter.encode(texts)
                    fake_lbs = torch.LongTensor(fake_lbs)
                    fake_lb_lens = torch.IntTensor([len(texts[0])])
                else:
                    fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow, ).to(self.device)

                rand_z.sample_()
                rand_styles = rand_z.unsqueeze(1).repeat(1, ncol, 1).view(nrow * ncol, self.opt.GenModel.style_dim)
                gen_imgs = self.models.G(rand_styles, fake_lbs, fake_lb_lens)
                enc_styles = self.models.E(gen_imgs, fake_lb_lens * self.opt.char_width, self.models.B)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    for j in range(ncol):
                        ax = plt.subplot(nrow, ncol, i * ncol + 1 + j)
                        gen_img = gen_imgs[i * ncol + j]
                        ax.imshow(gen_img, cmap='gray')
                        ax.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_text(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset(self.opt.dataset, self.opt.training.dset_split, process_style=True),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        def get_space_index(text):
            idxs = []
            for i, ch in enumerate(text):
                if ch == ' ':
                    idxs.append(i)
            return idxs

        with torch.no_grad():
            nrow = self.opt.test.nrow
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                batch = next(iter(tst_loader))
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device), batch['style_img_lens'].to(self.device)
                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.B)

                real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127
                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                space_indexs = get_space_index(text)
                for idx in space_indexs:
                    gen_imgs[:, :, idx * self.opt.char_width: (idx + 1) * self.opt.char_width] = -1
                gen_imgs, gen_img_lens = rescale_images2(gen_imgs, fake_lb_lens * self.opt.char_width, fake_lb_lens,
                                           batch['org_img_lens'].to(self.device),
                                           batch['lb_lens'].to(self.device))
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()

                for i in range(nrow):
                    plt.subplot(nrow * 2, 1, i * 2 + 1)
                    plt.imshow(real_imgs[i, :, :real_img_lens[i]], cmap='gray')
                    plt.axis('off')
                    plt.subplot(nrow * 2, 1, i * 2 + 2)
                    plt.imshow(gen_imgs[i, :, :gen_img_lens[i]], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()


class GlobalLocalAdversarialModel(AdversarialModel):
    def __init__(self, opt, log_root='./'):
        super(GlobalLocalAdversarialModel, self).__init__(opt, log_root)

        device = self.device

        generator = Generator(**opt.GenModel).to(device)
        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        patch_discriminator = PatchDiscriminator(**opt.PatchDiscModel).to(device)
        recognizer = Recognizer(**opt.OcrModel).to(device)

        self.models = Munch(
            G=generator,
            D=discriminator,
            P=patch_discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier,
            B=style_backbone,
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')
        self.classify_loss = CrossEntropyLoss()
        self.contextual_loss = CXLoss()

    def train(self):
        self.info()

        opt = self.opt
        self.z = prepare_z_dist(opt.training.batch_size, opt.EncModel.style_dim, self.device,
                                seed=self.opt.seed)
        self.y = prepare_y_dist(opt.training.batch_size, len(self.lexicon), self.device, seed=self.opt.seed)

        self.eval_z = prepare_z_dist(opt.training.eval_batch_size, opt.EncModel.style_dim, self.device,
                                     seed=self.opt.seed)
        self.eval_y = prepare_y_dist(opt.training.eval_batch_size, len(self.lexicon), self.device,
                                     seed=self.opt.seed)

        self.optimizers = Munch(
            G=torch.optim.Adam(chain(self.models.G.parameters(), self.models.E.parameters()),
                               lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(chain(self.models.D.parameters(), self.models.P.parameters()),
                               lr=opt.training.lr,
                               betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training),
            D=get_scheduler(self.optimizers.D, opt.training)
        )

        epoch_done = 1
        if os.path.exists(self.opt.training.pretrained_ckpt):
            epoch_done = self.load(self.opt.training.pretrained_ckpt, self.device)
            self.validate(style_guided=True)
        else:
            if os.path.exists(self.opt.training.pretrained_w):
                w_dict = torch.load(self.opt.training.pretrained_w, self.device)
                self.models.W.load_state_dict(w_dict['WriterIdentifier'])
                self.models.B.load_state_dict(w_dict['StyleBackbone'])
                print('load pretrained writer_identifier: ', self.opt.training.pretrained_w)
                # self.validate_wid()
            if os.path.exists(self.opt.training.pretrained_r):
                r_dict = torch.load(self.opt.training.pretrained_r)['Recognizer']
                self.models.R.load_state_dict(r_dict, self.device)
                print('load pretrained recognizer: ', self.opt.training.pretrained_r)
                # self.validate_ocr()

        # multi-gpu
        if self.local_rank > -1:
            for key in self.models.keys():
                self.models[key] = torch.nn.parallel.DistributedDataParallel(
                    self.models[key],
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False
                )

        self.averager_meters = AverageMeterManager(['adv_loss', 'fake_disc_loss',
                                                    'real_disc_loss', 'adv_loss_patch',
                                                    'fake_disc_loss_patch',
                                                    'real_disc_loss_patch', 'recn_loss',
                                                    'fake_ctc_loss', 'info_loss',
                                                    'fake_wid_loss', 'ctx_loss',
                                                    'kl_loss', 'gp_ctc', 'gp_info',
                                                    'gp_wid', 'gp_recn'])
        device = self.device

        if self.local_rank > -1:
            ctc_len_scale = self.models.R.module.len_scale
        else:
            ctc_len_scale = self.models.R.len_scale

        best_fid = np.inf
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs & Network Forward
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = batch['style_imgs'].to(device), \
                                                      batch['style_img_lens'].to(device), \
                                                      batch['wids'].to(device)
                real_aug_imgs, real_aug_img_lens = batch['aug_imgs'].to(device), batch['aug_img_lens'].to(device)
                real_lbs, real_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)
                max_label_len = real_lbs.size(-1)

                #############################
                # Optimizing Discriminator
                #############################
                self.optimizers.D.zero_grad()
                set_requires_grad([self.models.G, self.models.E, self.models.R, self.models.W, self.models.B], False)
                set_requires_grad([self.models.D, self.models.P], True)
                # self.models.B.frozen_bn()

                with torch.no_grad():
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                 self.opt.training.capitalize_ratio,
                                                 self.opt.training.blank_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    if self.vae_mode:
                        enc_z, _, _ = self.models.E(real_imgs, real_img_lens, self.models.B, vae_mode=True)
                    else:
                        enc_z = self.models.E(real_imgs, real_img_lens, self.models.B, vae_mode=False)

                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs, recn_imgs], dim=0)
                    cat_fake_lb_lens = torch.cat([fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0)
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                ### Compute discriminative loss for real & fake samples ###
                fake_disc = self.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
                fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))

                fake_img_patches = extract_all_patches(cat_fake_imgs, cat_fake_img_lens)
                fake_disc_patches = self.models.P(fake_img_patches.detach())
                fake_disc_loss_patch = torch.mean(F.relu(1.0 + fake_disc_patches))

                # real_imgs.requires_grad_()
                real_disc = self.models.D(real_imgs, real_img_lens, real_lb_lens)
                real_disc_aug = self.models.D(real_aug_imgs, real_aug_img_lens, real_lb_lens)
                real_disc_loss = (torch.mean(F.relu(1.0 - real_disc)) +
                                  torch.mean(F.relu(1.0 - real_disc_aug))) / 2

                real_img_patches = extract_all_patches(real_imgs, real_img_lens, plot=False)
                real_aug_imgs_patches = extract_all_patches(real_aug_imgs, real_aug_img_lens)
                real_disc_patches = self.models.P(torch.cat([real_img_patches,
                                                             real_aug_imgs_patches],
                                                             dim=0) .detach())
                real_disc_loss_patch = torch.mean(F.relu(1.0 - real_disc_patches))

                disc_loss = real_disc_loss + fake_disc_loss + real_disc_loss_patch + fake_disc_loss_patch
                self.averager_meters.update('real_disc_loss', real_disc_loss.item())
                self.averager_meters.update('fake_disc_loss', fake_disc_loss.item())
                self.averager_meters.update('real_disc_loss_patch', real_disc_loss_patch.item())
                self.averager_meters.update('fake_disc_loss_patch', fake_disc_loss_patch.item())

                disc_loss.backward()
                self.optimizers.D.step()

                #############################
                # Optimizing Generator
                #############################
                if iter_count % self.opt.training.num_critic_train == 0:
                    self.optimizers.G.zero_grad()
                    set_requires_grad([self.models.D, self.models.P, self.models.R, self.models.W, self.models.B], False)
                    set_requires_grad([self.models.G, self.models.E], True)
                    # self.models.B.frozen_bn()

                    ##########################
                    # Prepare Fake Inputs
                    ##########################
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                 self.opt.training.capitalize_ratio,
                                                 self.opt.training.blank_ratio,
                                                 sort=True)

                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    if self.vae_mode:
                        (enc_z, mu, logvar), real_img_feats = self.models.E(real_imgs, real_img_lens, self.models.B,
                                                                            ret_feats=True, vae_mode=True)
                    else:
                        enc_z, real_img_feats = self.models.E(real_imgs, real_img_lens, self.models.B,
                                                              ret_feats=True, vae_mode=False)
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                    ###################################################
                    # Calculating G Losses
                    ####################################################
                    ### deal with fake samples ###
                    ### Compute Adversarial loss ###
                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs, recn_imgs], dim=0)
                    cat_fake_lb_lens = torch.cat([fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0)
                    cat_fake_disc = self.models.D(cat_fake_imgs,
                                                  cat_fake_lb_lens * self.opt.char_width,
                                                  cat_fake_lb_lens)
                    adv_loss = -torch.mean(cat_fake_disc)

                    fake_img_patches = extract_all_patches(cat_fake_imgs, cat_fake_lb_lens * self.opt.char_width)
                    fake_disc_patches = self.models.P(fake_img_patches)
                    adv_loss_patch = -torch.mean(fake_disc_patches)

                    ### CTC Auxiliary loss ###
                    # self.models.R.frozen_bn()
                    fake_img_lens = fake_lb_lens * self.opt.char_width
                    fake_ctc_rand = self.models.R(fake_imgs, fake_img_lens)
                    fake_ctc_loss_rand = self.ctc_loss(fake_ctc_rand, fake_lbs,
                                                       fake_img_lens // ctc_len_scale,
                                                       fake_lb_lens)

                    style_img_lens = fake_lb_lens * self.opt.char_width
                    fake_ctc_style = self.models.R(style_imgs, style_img_lens)
                    fake_ctc_loss_style = self.ctc_loss(fake_ctc_style, fake_lbs,
                                                        style_img_lens // ctc_len_scale,
                                                        fake_lb_lens)

                    recn_img_lens = real_lb_lens * self.opt.char_width
                    fake_ctc_recn = self.models.R(recn_imgs, recn_img_lens)
                    fake_ctc_loss_recn = self.ctc_loss(fake_ctc_recn, real_lbs,
                                                       recn_img_lens // ctc_len_scale,
                                                       real_lb_lens)

                    fake_ctc_loss = fake_ctc_loss_rand + fake_ctc_loss_recn + fake_ctc_loss_style

                    ### Style Reconstruction ###
                    styles = self.models.E(fake_imgs, fake_lb_lens * self.opt.char_width, self.models.B)
                    info_loss = torch.mean(torch.abs(styles - self.z.detach()))

                    ### Content Restruction ###
                    recn_loss = recn_l1_loss(recn_imgs, real_imgs, real_img_lens)

                    ### Writer Identify Loss ###
                    cat_style_imgs = torch.cat([style_imgs, recn_imgs], dim=0)
                    cat_style_img_lens = torch.cat([fake_lb_lens, real_lb_lens], dim=0) * self.opt.char_width
                    recn_wid_logits, fake_imgs_feats = self.models.W(cat_style_imgs,
                                                                     cat_style_img_lens,
                                                                     self.models.B,
                                                                     ret_feats=True)
                    fake_wid_loss = self.classify_loss(recn_wid_logits, real_wids.repeat(2))

                    ###  Contextual Loss and Gram Loss for non-aligned data  ###
                    ctx_loss = torch.FloatTensor([0.]).to(self.device)
                    for real_img_feat, fake_img_feat \
                            in zip(real_img_feats, fake_imgs_feats):
                        fake_feat = fake_img_feat.chunk(2, dim=0)
                        # ctx_loss for style_imgs
                        ctx_loss += self.contextual_loss(real_img_feat, fake_feat[0])
                        # ctx_loss for recn_imgs
                        ctx_loss += self.contextual_loss(real_img_feat, fake_feat[1])

                    ### KL-Divergency loss ###
                    kl_loss = KLloss(mu, logvar) if self.vae_mode else torch.FloatTensor([0.]).to(self.device)

                    grad_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_OCR = torch.autograd.grad(fake_ctc_loss_rand, fake_ctc_rand, create_graph=True, retain_graph=True)[0]
                    grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=True, retain_graph=True)[0]
                    grad_fake_recn = torch.autograd.grad(recn_loss, enc_z, create_graph=True, retain_graph=True)[0]

                    std_grad_adv = torch.std(grad_fake_adv)
                    gp_ctc = torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1
                    gp_ctc.clamp_max_(100)
                    gp_info = torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1
                    gp_wid = torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1
                    gp_wid.clamp_max_(10)
                    gp_recn = torch.div(std_grad_adv, torch.std(grad_fake_recn) + 1e-8).detach() + 1

                    self.averager_meters.update('gp_ctc', gp_ctc.item())
                    self.averager_meters.update('gp_info', gp_info.item())
                    self.averager_meters.update('gp_wid', gp_wid.item())
                    self.averager_meters.update('gp_recn', gp_recn.item())

                    g_loss = adv_loss + adv_loss_patch +\
                             gp_ctc * fake_ctc_loss + \
                             gp_info * info_loss + \
                             gp_wid * fake_wid_loss + \
                             gp_recn * recn_loss + \
                             self.opt.training.lambda_ctx * ctx_loss + \
                             self.opt.training.lambda_kl * kl_loss
                    g_loss.backward()
                    self.averager_meters.update('adv_loss', adv_loss.item())
                    self.averager_meters.update('adv_loss_patch', adv_loss_patch.item())
                    self.averager_meters.update('fake_ctc_loss', fake_ctc_loss.item())
                    self.averager_meters.update('info_loss', info_loss.item())
                    self.averager_meters.update('fake_wid_loss', fake_wid_loss.item())
                    self.averager_meters.update('recn_loss', recn_loss.item())
                    self.averager_meters.update('ctx_loss', ctx_loss.item())
                    self.averager_meters.update('kl_loss', kl_loss.item())
                    self.optimizers.G.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    meter_vals = self.averager_meters.eval_all()
                    self.averager_meters.reset_all()
                    info = "[%3d|%3d]-[%4d|%4d] G:%.4f G-p:%.4f D-fake:%.4f D-real:%.4f " \
                           "D-fake-p:%.4f D-real-p:%.4f CTC-fake:%.4f Wid-fake:%.4f " \
                           "Recn-z:%.4f Recn-c:%.4f Ctx:%.4f Kl:%.4f" \
                           % (epoch, self.opt.training.epochs,
                              iter_count % len(self.train_loader), len(self.train_loader),
                              meter_vals['adv_loss'], meter_vals['adv_loss_patch'],
                              meter_vals['fake_disc_loss'], meter_vals['real_disc_loss'],
                              meter_vals['fake_disc_loss_patch'], meter_vals['real_disc_loss_patch'],
                              meter_vals['fake_ctc_loss'], meter_vals['fake_wid_loss'], meter_vals['info_loss'],
                              meter_vals['recn_loss'], meter_vals['ctx_loss'], meter_vals['kl_loss'])
                    self.print(info) if self.local_rank < 1 else None

                    if self.writer:
                        for key, val in meter_vals.items():
                            self.writer.add_scalar('loss/%s' % key, val,
                                                   iter_count + 1) if self.local_rank < 1 else None

                        try:
                            lr = self.lr_schedulers.G.get_last_lr()[0]
                        except Exception:
                            lr = self.lr_schedulers.G.get_lr()[0]
                        self.writer.add_scalar('loss/lr', lr, iter_count + 1) if self.local_rank < 1 else None

                        info_attns = self.models.G._info_attention()
                        for i_, info in enumerate(info_attns):
                            self.writer.add_scalar('loss/gamma%d' % i_, info['gamma'],
                                                   iter_count + 1) if self.local_rank < 1 else None
                            heatmap = plot_heatmap(info['out'])
                            self.writer.add_image('attention/%d' % i_, heatmap.transpose((2, 0, 1)))

                if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                    if not (self.logger and self.writer):
                        self.create_logger() if self.local_rank < 1 else None

                    sample_root = os.path.join(self.log_root, self.opt.training.sample_dir)
                    if not os.path.exists(sample_root):
                        os.makedirs(sample_root) if self.local_rank < 1 else None
                    self.sample_images(iter_count + 1) if self.local_rank < 1 else None

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        epoch % self.opt.training.save_epoch_val == 0:
                    self.print('Calculate FID_KID') if self.local_rank < 1 else None
                    scores = self.validate()

                    if 'fid' in scores and scores['fid'] < best_fid:
                        best_fid = scores['fid']
                        self.save('best', epoch, **scores) if self.local_rank < 1 else None

                    if self.writer:
                        for key, val in scores.items():
                            self.writer.add_scalar('valid/%s' % key, val, epoch) if self.local_rank < 1 else None

                if self.local_rank > -1:
                    dist.barrier()

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)


class RecognizeModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(RecognizeModel, self).__init__(opt, log_root)

        device = self.device
        self.collect_fn = get_collect_fn(sort_input=opt.training.sort_input, sort_style=False)
        recognizer = Recognizer(**opt.OcrModel).to(device)
        # print(recognizer.cnn_backbone)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device)['Recognizer']
            new_ckpt = {}
            for key, val in ckpt.items():
                if not key.startswith('ctc_cls'):
                    new_ckpt[key] = val
            recognizer.load_state_dict(new_ckpt, strict=False)
            print('load pretrained backbone from ', opt.training.pretrained_backbone)

        if os.path.exists(opt.training.resume):
            ckpt = torch.load(opt.training.resume, device)['Recognizer']
            recognizer.load_state_dict(ckpt)
            print('load pretrained model from ', opt.training.resume)

        self.models = Munch(R=recognizer)

        self.tst_loader = DataLoader(
            get_dataset(self.opt.valid.dset_name, self.opt.valid.dset_split, process_style=True),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=True, sort_style=True)
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')

    def train(self):
        self.info()

        trainset_info = (self.opt.training.dset_name, self.opt.training.dset_split, False, self.opt.training.augment, True)
        self.print('Trainset: {} [{}]'.format(*trainset_info))
        self.train_loader = DataLoader(
            get_dataset(*trainset_info),
            batch_size=self.opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4
        )

        self.optimizers = Munch(R=torch.optim.Adam(self.models.R.parameters(), lr=self.opt.training.lr))
        self.lr_schedulers = Munch(R=get_scheduler(self.optimizers.R, self.opt.training))

        epoch_done = 1
        if self.opt.training.resume:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        device = self.device
        ctc_loss_meter = AverageMeter()
        ctc_len_scale = self.models.R.len_scale
        best_cer = np.inf
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens = batch['aug_imgs'].to(device), batch['aug_img_lens'].to(device)
                real_lbs, real_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.R.zero_grad()
                ### Compute CTC loss for real samples###
                real_ctc = self.models.R(real_imgs, real_img_lens)
                real_ctc_lens = real_img_lens // ctc_len_scale
                real_ctc_loss = self.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
                ctc_loss_meter.update(real_ctc_loss.item())
                real_ctc_loss.backward()
                self.optimizers.R.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not (self.logger and self.writer):
                            self.create_logger()

                    try:
                        lr = self.lr_schedulers.R.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.R.get_lr()[0]

                    ctc_loss_avg = ctc_loss_meter.eval()
                    ctc_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] CTC: %.5f  Lr: %.6f" \
                           % (epoch, self.opt.training.epochs, iter_count % len(self.train_loader),
                              len(self.train_loader), ctc_loss_avg, lr)
                    self.print(info)

                    if self.writer:
                        self.writer.add_scalar('loss/ctc_loss', ctc_loss_avg, iter_count + 1)
                        self.writer.add_scalar('loss/lr', lr, iter_count + 1)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        (epoch % self.opt.training.save_epoch_val == 0 or
                         epoch >= self.opt.training.epochs):
                    self.print('Calculate CER_WER')
                    ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                    if not os.path.exists(ckpt_root):
                        os.makedirs(ckpt_root)

                    scores = self.validate()
                    wer, cer = scores['WER'], scores['CER']
                    self.print('WER:{} CER:{}'.format(wer, cer))
                    if cer < best_cer:
                        best_cer = cer
                        self.save('best', epoch, WER=wer, CER=cer)
                    if self.writer:
                        self.writer.add_scalar('valid/WER', wer, epoch)
                        self.writer.add_scalar('valid/CER', cer, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def validate(self, *args, **kwargs):
        self.set_mode('eval')
        ctc_len_scale = self.models.R.len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0
        print(self.tst_loader.dataset.file_path)
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.tst_loader), total=len(self.tst_loader)):
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device), batch['style_img_lens'].to(self.device)
                logits = self.models.R(real_imgs, real_img_lens)
                logits = torch.nn.functional.softmax(logits, dim=2).detach()

                logits = logits.cpu().numpy()
                word_preds = []
                for logit, img_len in zip(logits, batch['style_img_lens'].cpu().numpy()):
                    label = ctc_greedy_decoder(logit[:img_len // ctc_len_scale])
                    word_preds.append(self.label_converter.decode(label))

                word_reals = self.label_converter.decode(batch['lbs'], batch['lb_lens'])

                for word_pred, word_real in zip(word_preds, word_reals):
                    char_tran = levenshtein(word_pred, word_real)
                    char_trans += char_tran
                    total_chars += len(word_real)
                    total_words += 1
                    if char_tran > 0:
                        word_trans += 1

        for model in self.models.values():
            model.train()

        cer = char_trans * 1.0 / total_chars
        wer = word_trans * 1.0 / total_words
        return {'CER': cer, 'WER': wer}


class WriterIdentifyModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(WriterIdentifyModel, self).__init__(opt, log_root)

        device = self.device

        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device)

            if 'Recognizer' in ckpt:
                ckpt = ckpt['Recognizer']
                new_ckpt = {}
                for key, val in ckpt.items():
                    if key.startswith('cnn_backbone') or key.startswith('cnn_ctc'):
                        new_ckpt[key] = val
                style_backbone.load_state_dict(new_ckpt)
            else:
                ckpt = ckpt['StyleBackbone']
                style_backbone.load_state_dict(ckpt)

            print('Load style_backbone from ', opt.training.pretrained_backbone)

        identifier = WriterIdentifier(**opt.WidModel).to(device)
        self.models = Munch(W=identifier, B=style_backbone)

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=False)
        )

        self.wid_loss = CrossEntropyLoss()

    def train(self):
        self.info()

        trainset_info = (self.opt.training.dset_name,
                         self.opt.training.dset_split,
                         self.opt.training.random_clip,
                         False, self.opt.training.process_style)
        self.print('Trainset: {} [{}]'.format(*trainset_info))
        self.train_loader = DataLoader(
            get_dataset(*trainset_info),
            batch_size=self.opt.training.batch_size,
            shuffle=True,
            collate_fn=get_collect_fn(sort_input=True, sort_style=False),
            num_workers=4
        )

        if self.opt.training.frozen_backbone:
            print('frozen_backbone')
            self.optimizers = Munch(W=torch.optim.Adam(self.models.W.parameters()), lr=self.opt.training.lr)
        else:
            self.optimizers = Munch(W=torch.optim.Adam(
                                        chain(self.models.W.parameters(), self.models.B.parameters()),
                                    lr=self.opt.training.lr))

        self.lr_schedulers = Munch(W=get_scheduler(self.optimizers.W, self.opt.training))

        epoch_done = 1
        if self.opt.training.resume:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        device = self.device
        wid_loss_meter = AverageMeter()
        best_wrr = 0
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = batch['aug_imgs'].to(device), \
                                                      batch['aug_img_lens'].to(device), \
                                                      batch['wids'].to(device)

                if self.opt.training.frozen_backbone:
                    self.models.B.frozen_bn()

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.W.zero_grad()
                ### Compute CTC loss for real samples###
                wid_logits = self.models.W(real_imgs, real_img_lens, self.models.B)
                wid_loss = self.wid_loss(wid_logits, real_wids)
                wid_loss_meter.update(wid_loss.item())
                wid_loss.backward()
                self.optimizers.W.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not (self.logger and self.writer):
                            self.create_logger()

                    try:
                        lr = self.lr_schedulers.W.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.W.get_lr()[0]

                    wid_loss_avg = wid_loss_meter.eval()
                    wid_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] WID: %.5f  Lr: %.6f" \
                           % (epoch, self.opt.training.epochs, iter_count % len(self.train_loader),
                              len(self.train_loader), wid_loss_avg, lr)
                    self.print(info)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        (epoch % self.opt.training.save_epoch_val == 0 or
                         epoch >= self.opt.training.epochs):
                    self.print('Calculate WRR')
                    ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                    if not os.path.exists(ckpt_root):
                        os.makedirs(ckpt_root)

                    wrr = self.validate()
                    self.print('WRR:{:.2f}'.format(wrr))
                    if wrr > best_wrr:
                        best_wrr = wrr
                        self.save('best', epoch, WRR=wrr)
                    if self.writer:
                        self.writer.add_scalar('valid/WRR', wrr, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def validate(self, *args, **kwargs):
        self.set_mode('eval')

        with torch.no_grad():
            acc_counts = 0.
            total_counts = 0.
            for i, batch in tqdm(enumerate(self.tst_loader), total=len(self.tst_loader)):
                wid_logits = self.models.W(batch['style_imgs'].to(self.device),
                                           batch['style_img_lens'].to(self.device),
                                           self.models.B)
                _, preds = torch.max(wid_logits.data, dim=1)

                acc_counts += preds.eq(batch['wids'].to(self.device)).sum().item()
                total_counts += wid_logits.size(0)

            wrr = acc_counts * 100. / total_counts
            wier = 1 - acc_counts * 1. / total_counts
            print('wier: ', wier)

        for model in self.models.values():
            model.train()

        return wrr
