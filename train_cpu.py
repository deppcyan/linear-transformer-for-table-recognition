import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import utils
import commons
from models import TableRecognizer
from data_utils import (
    ImageTextLoader,
    ImageTextCollate,
)
from losses import (
    loss_fn_img,
    loss_fn_txt
)

global_step = 0


@hydra.main(config_path='configs/', config_name='linear_transformer')
def main(hps):
    """Assume Single Node Multi GPUs Training Only"""
    # assert torch.cuda.is_available(), "CPU training is not allowed."
    assert OmegaConf.select(hps, "model_dir") != ".hydra", "Please specify model_dir."
    print(OmegaConf.to_yaml(hps))
    run(hps)


def run(hps):
    global global_step
    writer = SummaryWriter(log_dir='./')
    writer_eval = SummaryWriter(log_dir='./eval')

    train_dataset = ImageTextLoader(hps.data.training_file_path, hps.data)
    collate_fn = ImageTextCollate()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn)

    eval_dataset = ImageTextLoader(hps.data.validation_file_path, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False, pin_memory=False,
                             collate_fn=collate_fn)

    model = TableRecognizer(
        len(train_dataset.vocab),
        3 * (hps.data.patch_length ** 2),
        **hps.model)
    optim = torch.optim.Adam(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path('./', "model_*.pth"), model, optim)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(0, epoch, hps, model, optim, scaler, [train_loader, eval_loader],
                           [writer, writer_eval])


def train_and_evaluate(epoch, hps, model, optim, scaler, loaders, writers):
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    model.train()
    for batch_idx, (img, txt, mask_img, mask_txt, pos_r, pos_c, pos_t) in enumerate(train_loader):
        img_i = img[:, :-1]
        img_o = img[:, 1:]
        txt_i = txt[:, :-1]
        txt_o = txt[:, 1:]
        mask_img_i = mask_img[:, :-1]
        mask_img_o = mask_img[:, 1:]
        mask_txt_i = mask_txt[:, :-1]
        mask_txt_o = mask_txt[:, 1:, 0]

        with autocast(enabled=hps.train.fp16_run):
            logits_img, logits_txt = model(img_i, txt_i, mask_img_i, mask_txt_i, pos_r, pos_c, pos_t)
            with autocast(enabled=False):
                loss_img = loss_fn_img(logits_img, img_o, mask_img_o)
                loss_txt = loss_fn_txt(logits_txt, txt_o, mask_txt_o)
                loss_tot = loss_img * hps.train.lamb + loss_txt
        optim.zero_grad()
        scaler.scale(loss_tot).backward()
        scaler.unscale_(optim)
        grad_norm = commons.grad_norm(model.parameters())
        scaler.step(optim)
        scaler.update()

        num_tokens = mask_img.sum() + mask_txt.sum()
        if global_step % hps.train.log_interval == 0:
            lr = optim.param_groups[0]['lr']
            losses = [loss_tot, loss_img, loss_txt]
            print('Train Epoch: {} [{:.0f}%]'.format(
                epoch,
                100. * batch_idx / len(train_loader)))
            print([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {"loss/total": loss_tot, "loss/img": loss_img, "loss/txt": loss_txt}
            scalar_dict.update({"learning_rate": lr, "grad_norm": grad_norm, "num_tokens": num_tokens})

            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars=scalar_dict)

        if global_step % hps.train.eval_interval == 0:
            print("START: EVAL")
            eval_loader.batch_sampler.set_epoch(global_step)
            evaluate(hps, model, eval_loader, writer_eval)
            utils.save_checkpoint(model, optim, hps.train.learning_rate, epoch,
                                  "model_{}.pth".format(global_step)
                                  )
            print("END: EVAL")
        global_step += 1

    print('====> Epoch: {}'.format(epoch))


def evaluate(hps, model, eval_loader, writer_eval):
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, txt, mask_img, mask_txt, pos_r, pos_c, pos_t) in enumerate(eval_loader):
            img_i = img[:, :-1]
            img_o = img[:, 1:]
            txt_i = txt[:, :-1]
            txt_o = txt[:, 1:]
            mask_img_i = mask_img[:, :-1]
            mask_img_o = mask_img[:, 1:]
            mask_txt_i = mask_txt[:, :-1]
            mask_txt_o = mask_txt[:, 1:, 0]

            with autocast(enabled=hps.train.fp16_run):
                logits_img, logits_txt = model(img_i, txt_i, mask_img_i, mask_txt_i, pos_r, pos_c, pos_t)
                with autocast(enabled=False):
                    loss_img = loss_fn_img(logits_img, img_o, mask_img_o)
                    loss_txt = loss_fn_txt(logits_txt, txt_o, mask_txt_o)
                    loss_tot = loss_img * hps.train.lamb + loss_txt
            break

    scalar_dict = {"loss/total": loss_tot.item(), "loss/img": loss_img.item(), "loss/txt": loss_txt.item()}

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict,
    )
    model.train()


if __name__ == "__main__":
    main()
