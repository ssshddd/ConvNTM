from curses.ascii import BS
from re import U
from tkinter import W
import torch.nn as nn
from torch.nn import functional as F
from utils.statistics import LossStatistics
from utils.time_log import time_since, convert_time2str
import time
import math
import logging
import torch
import sys
import os
import pandas as pd
import numpy as np

EPS = 1e-6


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


def l1_penalty(para):
    return nn.L1Loss(size_average=False)(para, torch.zeros_like(para))


def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def update_rec_M(cur_w_rec_M, cur_rec_M, target_rec_M):
    diff = (cur_rec_M-target_rec_M)/10
    cur_w_rec_M.mul_(2.0 ** diff)

def train_ntm_one_epoch(model, dataloader, optimizer, opt, epoch, uttr_reps=None):
    model.train()
    train_loss = 0
    train_rec_loss, train_kl = 0, 0
    for batch_idx, (data_bow, lens) in enumerate(dataloader):
        data_bow = data_bow.to(opt.device)
        optimizer.zero_grad()

        batch, recon_batch, mu, logvar = model(data_bow, lens, uttr_reps)
        rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
        loss = rec_loss + opt.w_KL * kl + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        train_loss += loss.item()
        train_rec_loss += rec_loss.item()
        train_kl += kl.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, Rec loss: {:.4f}, KL: {:.4f}'.format(
                   epoch, batch_idx * len(data_bow), len(dataloader.dataset),
                   100. * batch_idx / len(dataloader),
                   loss.item() / len(data_bow), 
                   rec_loss.item() / len(data_bow), 
                   kl.item() / len(data_bow)))

    logging.info('====>Train epoch: {} Average loss: {:.4f}, Rec loss: {:.4f}, KL: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset), train_rec_loss / len(dataloader.dataset), train_kl / len(dataloader.dataset)))
    sparsity = check_sparsity(model.fcd1.weight.data)

    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % opt.target_sparsity)
    update_l1(model.l1_strength, sparsity, opt.target_sparsity)
    return sparsity


def test_ntm_one_epoch(model, dataloader, opt, epoch, uttr_reps=None):
    model.eval()
    test_loss = 0
    test_rec_loss, test_kl = 0, 0
    with torch.no_grad():
        for i, (data_bow, lens) in enumerate(dataloader):
            data_bow = data_bow.to(opt.device)
            batch, recon_batch, mu, logvar = model(data_bow, lens, uttr_reps)
            rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
            test_loss += rec_loss.item() + opt.w_KL * kl.item()
            test_rec_loss += rec_loss.item()
            test_kl += kl.item()

    avg_loss = test_loss / len(dataloader.dataset)
    avg_rec_loss = test_rec_loss / len(dataloader.dataset)
    avg_kl = test_kl / len(dataloader.dataset)
    logging.info('====> Test epoch: {} Average loss: {:.4f}, Rec loss: {:.4f}, KL loss: {:.4f}'.format(epoch, avg_loss, avg_rec_loss, avg_kl))
    return avg_loss


def fix_model(model):
    for name, param in model.named_parameters():
        if name == 'fcd1.weight':
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True

def unfix_model_1(model):
    for name, param in model.named_parameters():
        if name != 'fcd1.weight':
            param.requires_grad = True
        else:
            param.requires_grad = False


def out_topic_label(model, ntm_model, train_data_loader, valid_data_loader, test_data_loader, bow_dictionary, 
                    train_bow_loader, valid_bow_loader, test_bow_loader, opt):
    if opt.load_pretrain_ntm:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))
        model.load_state_dict(torch.load(opt.check_pt_model_path))
    ntm_model.eval()
    model.eval()

    train_topic_idx, valid_topic_idx, test_topic_idx = [], [], []
    with torch.no_grad():
        rec_loss_total, kl_loss_total = 0, 0
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(train_data_loader , train_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()
            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)
            assert lens[0] == g.shape[0]
            topic_idx = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())
            train_topic_idx.append(topic_idx)
            rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl.item()
            if batch_i % 500 == 0 and batch_i > 0:
                print(batch_i, rec_loss_total/batch_i, kl_loss_total/batch_i)
        print(rec_loss_total/len(train_data_loader), kl_loss_total/len(train_data_loader))

        rec_loss_total, kl_loss_total = 0, 0
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(valid_data_loader , valid_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()
            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)
            assert lens[0] == g.shape[0]
            topic_idx = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())
            valid_topic_idx.append(topic_idx)
            rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl.item()
            if batch_i % 500 == 0 and batch_i > 0:
                print(batch_i, rec_loss_total/batch_i, kl_loss_total/batch_i)
        print(rec_loss_total/len(valid_data_loader), kl_loss_total/len(valid_data_loader))

        rec_loss_total, kl_loss_total = 0, 0
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(test_data_loader , test_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()
            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)
            assert lens[0] == g.shape[0]
            topic_idx = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())
            test_topic_idx.append(topic_idx)
            rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl.item()
            if batch_i % 500 == 0 and batch_i > 0:
                print(batch_i, rec_loss_total/batch_i, kl_loss_total/batch_i)
        print(rec_loss_total/len(test_data_loader), kl_loss_total/len(test_data_loader))

    return train_topic_idx, valid_topic_idx, test_topic_idx


def out_topic_words(model, ntm_model, train_data_loader, valid_data_loader, test_data_loader, bow_dictionary, 
                    train_bow_loader, valid_bow_loader, test_bow_loader, opt):
    if opt.load_pretrain_ntm:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))
        model.load_state_dict(torch.load(opt.check_pt_model_path))
    ntm_model.eval()
    model.eval()

    feature_names = list(bow_dictionary.token2id.keys())

    train_topic_words, valid_topic_words, test_topic_words = [], [], []
    with torch.no_grad():
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(train_data_loader , train_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()

            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)

            assert lens[0] == g.shape[0]
            topic_idx = []
            topic_words = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())

            topic_idx = list(set(topic_idx))
            
            beta = ntm_model.get_beta()

            beta_idx = torch.argsort(beta[topic_idx, :].reshape(-1), descending=True)

            for idx in beta_idx:
                if feature_names[idx%len(feature_names)] not in topic_words:
                    topic_words.append(feature_names[idx%len(feature_names)])
                    if len(topic_words) == 100:
                        break
            
            train_topic_words.append(topic_words)
            if (batch_i+1) % 500 == 0:
                print(topic_words)

        
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(valid_data_loader , valid_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()

            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)

            assert lens[0] == g.shape[0]
            topic_idx = []
            topic_words = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())

            topic_idx = list(set(topic_idx))
            
            beta = ntm_model.get_beta()

            beta_idx = torch.argsort(beta[topic_idx, :].reshape(-1), descending=True)

            for idx in beta_idx:
                if feature_names[idx%len(feature_names)] not in topic_words:
                    topic_words.append(feature_names[idx%len(feature_names)])
                    if len(topic_words) == 100:
                        break
            
            valid_topic_words.append(topic_words)
            if (batch_i+1) % 500 == 0:
                print(topic_words)


        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(test_data_loader , test_bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()

            uttr_reps = model(batch_input)
            g, batch, recon_batch, mu, logvar = ntm_model(uttr_bow=batch_bow, lens=lens, uttr_reps=uttr_reps, out_g=True)

            assert lens[0] == g.shape[0]
            topic_idx = []
            topic_words = []
            for i in range(g.shape[0]):
                if i % 2 == 0:
                    topic_idx.append(g[i].argmax().item())
            for i in range(g.shape[0]):
                if i % 2 == 1:
                    topic_idx.append(g[i].argmax().item())

            topic_idx = list(set(topic_idx))
            
            beta = ntm_model.get_beta()

            beta_idx = torch.argsort(beta[topic_idx, :].reshape(-1), descending=True)

            for idx in beta_idx:
                if feature_names[idx%len(feature_names)] not in topic_words:
                    topic_words.append(feature_names[idx%len(feature_names)])
                    if len(topic_words) == 100:
                        break
            
            test_topic_words.append(topic_words)
            if (batch_i+1) % 500 == 0:
                print(topic_words)

    return train_topic_words, valid_topic_words, test_topic_words



def train_model(model, ntm_model, optimizer_ml, optimizer_ntm, optimizer_whole, train_data_loader, valid_data_loader,
                bow_dictionary, train_bow_loader, valid_bow_loader, opt):
    logging.info('======================  Start Training  =========================')

    feature_names = list(bow_dictionary.token2id.keys())
    data = pd.read_csv("processed_data/{}/convs_nostop.csv".format(opt.data_tag), header=0, dtype={'label': int, 'train': int})
    common_texts = [text for text in data['content'].values]

    if opt.only_train_ntm and not opt.load_pretrain_ntm:
        print("\nWarming up ntm for %d epochs" % opt.ntm_warm_up_epochs)
        for epoch in range(1, opt.ntm_warm_up_epochs + 1):
            sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, opt, epoch)
            val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, opt, epoch)

            logging.getLogger().setLevel(logging.WARNING)
            TD, cv, npmi = ntm_model.eval_topic(feature_names, common_texts=common_texts)
            logging.getLogger().setLevel(logging.INFO)
            for i, n in enumerate([5,10,15,20,25]):
                logging.info('top{}, TD: {}, cv: {:.6f}, npmi: {:.6f}'.format(n, TD[i], cv[i], npmi[i]))

            TD, cv, npmi = max(TD), max(cv), max(npmi)
            best_ntm_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.TD=%.3f.cv=%.3f.npmi=%.3fntm_model' %
                                                   (epoch, val_loss, sparsity, TD, cv, npmi))
            logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
            torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    elif opt.load_pretrain_ntm:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))
        model.load_state_dict(torch.load(opt.check_pt_model_path))
        if opt.only_eval:
            logging.getLogger().setLevel(logging.WARNING)
            TD, cv, npmi = ntm_model.eval_topic(feature_names, common_texts=common_texts)
            logging.getLogger().setLevel(logging.INFO)
            for i, n in enumerate([5,10,15,20,25]):
                logging.info('top{}, TD: {}, cv: {:.6f}, npmi: {:.6f}'.format(n, TD[i], cv[i], npmi[i]))
            return

    if opt.only_train_ntm:
        return

    total_batch = 0
    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss, best_td, best_cv, best_npmi = float('inf'),float('-inf'),float('-inf'),float('-inf')
    best_td1, best_cv1, best_npmi1 = float('-inf'),float('-inf'),float('-inf')
    best_ntm_valid_loss = float('inf')
    joint_train_patience = 1
    ntm_train_patience = 1
    global_patience = 5
    num_stop_dropping = 0
    num_stop_dropping_ntm = 0
    num_stop_dropping_global = 0

    t0 = time.time()
    # Train_Seq2seq = True
    begin_iterate_train_ntm = opt.iterate_train_ntm
    check_pt_model_path = ""
    print("\nEntering main training for %d epochs" % opt.epochs)
    if opt.warmup:
        print('using warmup lr')
        warmup_epoch = int(opt.epochs * 0.05)
        iter_per_epoch = len(train_data_loader)
        warm_up_with_cosine_lr = lambda epoch: epoch / (warmup_epoch * iter_per_epoch) if epoch <= (warmup_epoch * iter_per_epoch) else 0.5 * (math.cos((epoch - warmup_epoch * iter_per_epoch) / ((opt.epochs - warmup_epoch) * iter_per_epoch) * math.pi) + 1)
        opt.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_whole, lr_lambda=warm_up_with_cosine_lr)

    
    for epoch in range(1, opt.epochs + 1):
        # if Train_Seq2seq:
        if epoch <= opt.pre_enc or not opt.joint_train:
            optimizer = optimizer_ml
            model.train()
            ntm_model.eval()
            logging.info("\nTraining encoder epoch: {}/{}".format(epoch, opt.epochs))
        elif begin_iterate_train_ntm:
            optimizer = optimizer_ntm
            model.train()
            ntm_model.train()
            fix_model(model)
            logging.info("\nTraining ntm epoch: {}/{}".format(epoch, opt.epochs))
            begin_iterate_train_ntm = False
        else:
            optimizer = optimizer_whole
            unfix_model(model)
            # unfix_model_1(ntm_model)
            model.train()
            ntm_model.train()
            logging.info("\nTraining encoder-ntm epoch: {}/{}".format(epoch, opt.epochs))
            if opt.iterate_train_ntm:
                begin_iterate_train_ntm = True


        logging.info("The total num of batches: %d, current learning rate:%.6f" %
                     (len(train_data_loader), optimizer.param_groups[0]['lr']))
        if opt.warmup:
            logging.info("current lr: %.6f" % opt.scheduler.get_last_lr()[0] )

        time_start_train = time.time()
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(train_data_loader , train_bow_loader)):
            total_batch += 1
            batch_loss_stat = train_one_batch(batch_input, batch_bow, lens, model, ntm_model, optimizer, opt, batch_i)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            if (batch_i + 1) % (len(train_data_loader) // 10) == 0:
                print("Train: %d/%d batches, current avg loss: %.3f, rec_loss: %.3f, kl: %.3f" %
                      ((batch_i + 1), len(train_data_loader), batch_loss_stat.loss / len(batch_bow),
                      batch_loss_stat.rec_loss / len(batch_bow), batch_loss_stat.kl / len(batch_bow)))

        time_train_one = time.time()-time_start_train
        current_train_loss = report_train_loss_statistics.loss / len(train_data_loader.dataset)
        current_train_rec_loss = report_train_loss_statistics.rec_loss / len(train_data_loader.dataset)
        current_train_kl = report_train_loss_statistics.kl / len(train_data_loader.dataset)
        current_train_rec_M = report_train_loss_statistics.rec_M / len(train_data_loader.dataset)

        
        logging.info('====>Train epoch: {} Average loss: {:.4f}, Rec loss: {:.4f}, KL: {:.4f}, time: {:.4f}s'.format(
            epoch, current_train_loss, current_train_rec_loss, current_train_kl, time_train_one))
        sparsity = check_sparsity(ntm_model.fcd1.weight.data)
        logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, ntm_model.l1_strength))
        logging.info("Target sparsity = %.3f" % opt.target_sparsity)
        update_l1(ntm_model.l1_strength, sparsity, opt.target_sparsity)

        # test the model on the validation dataset for one epoch
        model.eval()
        ntm_model.eval()
        time_start_infer = time.time()
        valid_loss_stat = evaluate_loss(valid_data_loader, valid_bow_loader, model, ntm_model, opt)
        time_infer_one = time.time()-time_start_infer
        current_valid_loss = valid_loss_stat.loss

        # debug
        if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
            logging.info(
                "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
            exit()

        logging.getLogger().setLevel(logging.WARNING)
        if True:
            TD, cv, npmi = ntm_model.eval_topic(feature_names, common_texts=common_texts)
        else:
            ntm_model.eval_topic(feature_names)
        logging.getLogger().setLevel(logging.INFO)
        for i, n in enumerate([5,10,15,20,25]):
            logging.info('top{}, TD: {}, cv: {:.6f}, npmi: {:.6f}'.format(n, TD[i], cv[i], npmi[i]))

        TD1, cv1, npmi1 = TD[1], cv[1], npmi[1]
        TD = sum(TD)/len(TD)
        cv = sum(cv)/len(cv)
        npmi = sum(npmi)/len(npmi)
        
        if current_valid_loss < best_valid_loss or TD>best_td or cv>best_cv or npmi>best_npmi or TD1>best_td1 or cv1>best_cv1 or npmi1>best_npmi1:  # update the best valid loss and save the model parameters
            print("Valid loss drops")
            sys.stdout.flush()
            best_valid_loss = current_valid_loss
            best_td = max(best_td, TD)
            best_cv = max(best_cv, cv)
            best_npmi = max(best_npmi, npmi)
            best_td1 = max(best_td1, TD1)
            best_cv1 = max(best_cv1, cv1)
            best_npmi1 = max(best_npmi1, npmi1)
            num_stop_dropping = 0
            num_stop_dropping_global = 0
            if epoch >= opt.start_checkpoint_at and not opt.save_each_epoch:
                check_pt_model_path = os.path.join(opt.model_path, 'e%d.bestval_loss=%.3f.td=%.3f(%.3f).cv=%.5f(%.5f).npmi=%.5f(%.5f).model-%s' %
                                                   (epoch, current_valid_loss, best_td, best_td1, best_cv, best_cv1, best_npmi, best_npmi1, convert_time2str(time.time() - t0)))
                # save model parameters
                torch.save(
                    model.state_dict(),
                    open(check_pt_model_path, 'wb')
                )
                logging.info('Saving encoder checkpoints to %s' % check_pt_model_path)

                if opt.joint_train:
                    check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                    # save model parameters
                    torch.save(
                        ntm_model.state_dict(),
                        open(check_pt_ntm_model_path, 'wb')
                    )
                    logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)
                if TD>best_td or cv>best_cv or npmi>best_npmi or TD1>best_td1 or cv1>best_cv1 or npmi1>best_npmi1:
                    best_check_pt_ntm_model_path = check_pt_ntm_model_path
                    best_check_pt_model_path = check_pt_model_path
        else:
            print("Valid loss does not drop")
            sys.stdout.flush()
            if (current_train_rec_M-opt.target_rec_M)/0.05 <= 1:
                num_stop_dropping += 1
                num_stop_dropping_global += 1
                # decay the learning rate by a factor
                for i, param_group in enumerate(optimizer.param_groups):
                    old_lr = float(param_group['lr'])
                    new_lr = old_lr * opt.learning_rate_decay
                    if old_lr - new_lr > EPS:
                        param_group['lr'] = new_lr
                        print("The new learning rate is decayed to %.6f" % new_lr)

        time_infer_one1 = time.time()-time_start_infer
        if opt.save_each_epoch:
            check_pt_model_path = os.path.join(opt.model_path, 'e%d.train_loss=%.3f.val_loss=%.3f.model-%s' %
                                               (epoch, current_train_loss, current_valid_loss,
                                                convert_time2str(time.time() - t0)))
            torch.save(  # save model parameters
                model.state_dict(),
                open(check_pt_model_path, 'wb')
            )
            logging.info('Saving encoder checkpoints to %s' % check_pt_model_path)


        logging.info('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
        logging.info('Epoch: %d; infer Time spent: %.2f (%.2f)' % (epoch, time_infer_one, time_infer_one1))

        logging.info(
            'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                current_train_loss, current_valid_loss, best_valid_loss))

        report_train_loss.append(current_train_loss)
        report_valid_loss.append(current_valid_loss)

        report_train_loss_statistics.clear()

        if not opt.save_each_epoch and num_stop_dropping >= opt.early_stop_tolerance:  # not opt.joint_train or
            logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)

            break

    model.load_state_dict(torch.load(best_check_pt_model_path))
    ntm_model.load_state_dict(torch.load(best_check_pt_ntm_model_path))
    valid_loss_stat = evaluate_loss(valid_data_loader, valid_bow_loader, model, ntm_model, opt)

    logging.getLogger().setLevel(logging.WARNING)
    TD, cv, npmi = ntm_model.eval_topic(feature_names, common_texts=common_texts)
    logging.getLogger().setLevel(logging.INFO)
    for i, n in enumerate([5,10,15,20,25]):
        logging.info('top{}, TD: {}, cv: {:.6f}, npmi: {:.6f}'.format(n, TD[i], cv[i], npmi[i]))

    return check_pt_model_path, check_pt_ntm_model_path


def train_one_batch(batch_input, batch_bow, lens, model, ntm_model, optimizer, opt, batch_idx):
    assert len(batch_input) == len(batch_bow) == len(lens)
    batch_bow = batch_bow.to(opt.device)
    optimizer.zero_grad()
    start_time = time.time()
    if opt.use_speaker_reps:
        uttr_reps = model(batch_input)
        batch, recon_batch, mu, logvar = ntm_model(batch_bow, lens, uttr_reps)
    else:
        batch, recon_batch, mu, logvar = ntm_model(batch_bow, lens)

    forward_time = time_since(start_time)
    start_time = time.time()
    rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)

    loss = opt.w_recbow * rec_loss + opt.w_KL * kl
    if opt.add_co_matirx:
        beta = ntm_model.get_beta()
        M = torch.load("processed_data/{}/co_matrix.pt".format(opt.data_tag)).to(beta.device)
        loss_rec_M = F.binary_cross_entropy(torch.matmul(beta.T,  beta), M, size_average=False)
        if loss_rec_M <= opt.target_rec_M:
            w_rec_M = 0
            unfix_model(model)
            unfix_model(ntm_model)
        else:
            tmp = (loss_rec_M-opt.target_rec_M)/0.05
            if tmp <= 1:
                unfix_model(model)
                unfix_model(ntm_model)
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = opt.learning_rate
            else:
                fix_model(model)
                fix_model(ntm_model)
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = opt.fast_lr
            w_rec_M = min(1, tmp)

        loss = (1-w_rec_M) * loss + w_rec_M * loss_rec_M
    else:
        loss_rec_M = torch.tensor([0.0])
    loss_compute_time = time_since(start_time)
    start_time = time.time()
    loss.div(len(batch_bow)).backward()
    backward_time = time_since(start_time)
    if opt.max_grad_norm > 0:
        if opt.use_speaker_reps:
            grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        grad_norm_before_clipping_ = nn.utils.clip_grad_norm_(ntm_model.parameters(), opt.max_grad_norm)
    optimizer.step()
    if opt.warmup:
        opt.scheduler.step()
    stat = LossStatistics(loss=loss.item(), rec_loss=rec_loss.item(), kl=kl.item(), rec_M=loss_rec_M.item(),
                          n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat

def evaluate_loss(data_loader, bow_loader, model, ntm_model, opt, eval='valid'):
    model.eval()
    ntm_model.eval()
    evaluation_loss_sum = 0.0
    evaluation_rec_loss = 0.0
    evaluation_kl = 0.0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    print("Evaluate loss for %d batches" % len(data_loader))
    with torch.no_grad():
        for batch_i, (batch_input, (batch_bow, lens)) in enumerate(zip(data_loader , bow_loader)):
            assert len(batch_input) == len(batch_bow) == len(lens)
            batch_bow = batch_bow.to(opt.device)
            start_time = time.time()
            if opt.use_speaker_reps:
                uttr_reps = model(batch_input)
                batch, recon_batch, mu, logvar = ntm_model(batch_bow, lens, uttr_reps)
            else:
                batch, recon_batch, mu, logvar = ntm_model(batch_bow, lens)
            forward_time = time_since(start_time)
            forward_time_total += forward_time
            start_time = time.time()
            rec_loss, kl = loss_function(recon_batch, batch, mu, logvar)
            loss = rec_loss + opt.w_KL * kl

            if opt.add_co_matirx:
                beta = ntm_model.get_beta()
                M = torch.load("processed_data/{}/co_matrix.pt".format(opt.data_tag)).to(beta.device)
                loss_rec_M = F.binary_cross_entropy(torch.matmul(beta.T,  beta), M, size_average=False)
                if loss_rec_M <= opt.target_rec_M:
                    w_rec_M = 0
                else:
                    w_rec_M = min(1, (loss_rec_M-opt.target_rec_M)/0.2)
                
                loss = (1-w_rec_M) * loss + w_rec_M * loss_rec_M
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time
            evaluation_loss_sum += loss.item()
            evaluation_rec_loss += rec_loss.item()
            evaluation_kl += kl.item()
            n_batch += len(batch_bow)
            if (batch_i + 1) % (len(data_loader) // 5) == 0:
                print("%s: %d/%d batches, current avg loss: %.3f, rec_loss: %.3f, kl: %.3f" %
                      (eval, (batch_i + 1), len(data_loader), loss.item() / len(batch_bow), rec_loss.item()/len(batch_bow), kl.item()/len(batch_bow)))

    avg_loss = evaluation_loss_sum / len(data_loader.dataset)
    avg_rec_loss = evaluation_rec_loss / len(data_loader.dataset)
    avg_kl = evaluation_kl / len(data_loader.dataset)
    logging.info('====> {} Average loss: {:.4f}, rec_loss: {:.4f}, kl: {:.4f}'.format(eval, avg_loss, avg_rec_loss, avg_kl))

    eval_loss_stat = LossStatistics(avg_loss, avg_rec_loss, avg_kl, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)    

    return eval_loss_stat