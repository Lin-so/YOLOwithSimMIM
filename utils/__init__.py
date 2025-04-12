import torch.nn as nn
import numpy as np
from SimMIM.simmim import SimMIM, SwinTransformerForSimMIM
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr
from ultralytics.nn.modules import Conv, C2f, Detect, C3k2, Concat

# 掩码生成器

class MaskGenerator:
    """
    为PatchEmbed后的图像生成掩码
    参数：
        input_size(int):原图像大小
        mask_patch_size(int):掩码分块大小
        model_patch_size(int):图像分块大小
        mask_radio(float):掩码率
    例如：
        input_size = 640, mask_patch_size = 32, model_patch_size = 4, mask_radio = 0.6
        mask的最小单位大小:rand_size = 640//32 = 20 
        最小单位额外重复的次数:scale = 20//4 = 5
        最小单位的像素的数量:token_count = 20**2 = 400
        随机生成的掩码的数量:mask_count = 400*0.6 = 240
        解释：
            mask的最小单位大小(20*20)横纵坐标分别额外重复5次,大小变为(160*160)
            而(640*640)的输入图像在model_patch_size下分块结果为(160*160)
    """
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        # 在0~token_count范围内随机打乱顺序，打乱后将前mask_count作为需要掩码的下标
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        # 生成长度为token_count的数组，并按照掩码下标掩码
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        # 将token_count数组变为rand_size*rand_size数组
        mask = mask.reshape((self.rand_size, self.rand_size))
        # 横纵坐标分别重复scale次
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    
def build_simmim(cfg):
    depth, width, max_channels = cfg["scales"][cfg.get("scale",'n')]
    depths = []
    for d in cfg['depths']:
        if d > 1:
            d = max(round(d*depth),1)
        depths.append(d)
    simmim = SimMIM(SwinTransformerForSimMIM(
        img_size = cfg['img_size'],
        patch_size = cfg['patch_size'],
        in_chans = 3,
        num_classes = 0,
        embed_dim = make_divisible(min(cfg['embed_dim'],max_channels)*width,8),
        depths = depths,
        num_heads = cfg['num_heads'],
        window_size = cfg['window_size']
        ),32)
    LOGGER.info(f"\n{colorstr('green','SimMIM BackBone Part'):^111}")
    for i, layer in enumerate(simmim.encoder.layers):
        p = sum(x.numel() for x in layer.parameters())  # number params
        LOGGER.info(f"{i:>3}{'-1':>20}{layer.depth:>3}{p:10.0f}  {'SwinTransformer':<45}{str([layer.dim,layer.input_resolution,layer.blocks[0].num_heads]):<30}")  # print
    return simmim

def build_yolohead(d):
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    embed_dim = d['embed_dim']
    depths = len(d['depths'])
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    LOGGER.info(f"\n{colorstr('green','YOLO Head Part'):^111}")
    LOGGER.info(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    embed_dim = make_divisible(min(embed_dim, max_channels) * width, 8)
    ch = [*[embed_dim*2**x for x in range(1,depths)]]
    ch.append(embed_dim*2**(depths-1))
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["head"], start=depths):  # from, number, module, args
        m = getattr(nn, m[3:]) if "nn." in m else globals()[m]  # get module
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {C2f, C3k2}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)  # number of repeats
            n = 1
        elif m is Conv:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args[0] = nc
            if len(args) < 2:
                args.append([ch[x] for x in f])
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)