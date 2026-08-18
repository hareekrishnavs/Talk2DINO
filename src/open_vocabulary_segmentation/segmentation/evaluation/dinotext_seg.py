import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_logger

from models.dinotext.cover_dr import (
    RWRInferenceConfig,
    RWRRuntimeSummary,
    apply_rwr_to_e3_snapshot,
)


class DINOTextSegInference(nn.Module):
    def __init__(
            self,
            model,
            text_embedding,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            rwr=None,
            # kp_w=0.3,
            **kwargs,
    ):
        super().__init__()

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy
        self.rwr_config = RWRInferenceConfig.from_mapping(rwr)
        self.rwr_runtime = RWRRuntimeSummary()
        # self.kp_w = kp_w

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        if self.rwr_config.enabled:
            if self.pamr:
                raise ValueError("canonical RWR evaluation does not support PAMR")
            if self.with_bg:
                raise ValueError(
                    "canonical RWR evaluation requires no background class"
                )
            if self.num_classes != self.rwr_config.expected_class_count:
                raise ValueError(
                    "canonical RWR evaluation class-count mismatch: "
                    f"expected {self.rwr_config.expected_class_count}, "
                    f"observed {self.num_classes}"
                )
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() != 1
            ):
                raise RuntimeError(
                    "canonical RWR reproduction requires exactly one process/GPU"
                )

        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building DINOTextSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}"
        )

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        if self.rwr_config.enabled:
            snapshot = self.model.generate_patch_snapshot(
                img,
                self.text_embedding,
            )
            rwr_output = apply_rwr_to_e3_snapshot(snapshot, self.rwr_config)
            self.rwr_runtime.add(rwr_output)
            masks = self.model.masks_from_patch_scores(
                rwr_output.patch_scores,
                snapshot.grid_hw,
                tuple(img.shape[-2:]),
            )
        else:
            masks, simmap = self.model.generate_masks(
                img,
                self.text_embedding,
                apply_pamr=self.pamr,
                # kp_w=self.kp_w,
            )

        B, N, H, W = masks.shape

        if self.with_bg:

            masks = masks.cpu()

            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        return masks

    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop = img[:, :, y1:y2, x1:x2]
                crop_logits = self.encode_decode(crop, img_meta)
                preds += F.pad(
                    crop_logits,
                    (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        if torch.any(count_mat == 0):
            raise RuntimeError("Sliding-window inference left uncovered pixels")
        preds = preds / count_mat
        return self._rescale(preds, img_meta) if rescale else preds

    def whole_inference(self, img, img_meta, rescale):
        logits = self.encode_decode(img, img_meta)
        return self._rescale(logits, img_meta) if rescale else logits

    def _rescale(self, logits, img_meta):
        resize_h, resize_w = img_meta[0]["img_shape"][:2]
        logits = logits[:, :, :resize_h, :resize_w]
        return F.interpolate(
            logits,
            size=img_meta[0]["ori_shape"][:2],
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def inference(self, img, img_meta, rescale=True):
        mode = self.test_cfg.get("mode", "whole")
        if mode == "slide":
            output = self.slide_inference(img, img_meta, rescale)
        elif mode == "whole":
            output = self.whole_inference(img, img_meta, rescale)
        else:
            raise ValueError(f"Unsupported test mode: {mode}")

        output = output.softmax(dim=1)
        if img_meta[0].get("flip", False):
            direction = img_meta[0].get("flip_direction")
            if direction == "horizontal":
                output = output.flip(dims=(3,))
            elif direction == "vertical":
                output = output.flip(dims=(2,))
            else:
                raise ValueError(f"Unsupported flip direction: {direction}")
        return output

    def simple_test(self, img, img_meta, rescale=True):
        prediction = self.inference(img, img_meta, rescale).argmax(dim=1)
        return list(prediction.cpu().numpy())

    def aug_test(self, imgs, img_metas, rescale=True):
        if not rescale:
            raise ValueError("Augmented inference requires rescale=True")
        logits = self.inference(imgs[0], img_metas[0], rescale)
        for index in range(1, len(imgs)):
            logits += self.inference(imgs[index], img_metas[index], rescale)
        prediction = (logits / len(imgs)).argmax(dim=1)
        return list(prediction.cpu().numpy())

    def forward(self, img, img_metas, return_loss=False, rescale=True, **kwargs):
        if return_loss:
            raise RuntimeError("DINOTextSegInference is evaluation-only")
        if not isinstance(img, list) or not isinstance(img_metas, list):
            raise TypeError("Evaluation inputs must be augmentation lists")
        if len(img) == 1:
            return self.simple_test(img[0], img_metas[0], rescale=rescale)
        return self.aug_test(img, img_metas, rescale=rescale)
