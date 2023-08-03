from .base import *
# from .eval import val_result
from loss_funcs import Loss, Learnable_Loss
from crowd3d.lib.utils.train_utils import test_train_fix_params, test_train_fix_bn, test_train_fix_rules

np.set_printoptions(precision=2, suppress=True)

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self.set_gpus()
        self._build_model_()
        self._build_optimizer()
        # self.set_up_val_loader()
        self._calc_loss = Loss()
        self.loader = self._create_data_loader(train_flag=True)
        self.mutli_task_uncertainty_weighted_loss = Learnable_Loss(self.loader.dataset._get_ID_num_()).cuda()
        self.optimizer.add_param_group({'params': self.mutli_task_uncertainty_weighted_loss.parameters()})

        self.train_cfg = {'mode': 'matching_gts', 'is_training': True, 'update_data': True,
                          'calc_loss': True if self.model_return_loss else False, \
                          'new_training': True if args().new_training_epoch>0 else False}
        self.val_best_PAMPJPE = {'pw3d': 60, 'mpiinf': 80}
        logging.info('Initialization of Trainer finished!')

    def set_gpus(self):
        gpu_list=args().gpu
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x).strip() for x in gpu_list)
    
    def train(self):
        # init_seeds(self.local_rank, cuda_deterministic=False)
        logging.info('start training')
        self.model.train()
        

        if self.fix_backbone_training_scratch:
            fix_backbone(self.model, exclude_key=['backbone.'])
        else:
            train_entire_model(self.model)
        
        if len(self.finetune_mode) > 0:
            keep_keys=test_train_fix_rules[self.finetune_mode]
            test_train_fix_bn(self.model, keep_key_list=keep_keys)
            test_train_fix_params(self.model, keep_key_list=keep_keys)
            print('++++++++fix_test_train++++++++++++', keep_keys)

        for epoch in range(self.epoch):
            # if epoch == 0: #orign:1. don't know why write 1 of ROMP.
            #     train_entire_model(self.model)
            self.train_epoch(epoch)
        self.summary_writer.close()

    def train_step(self, meta_data):
        self.optimizer.zero_grad()
        outputs = self.network_forward(self.model, meta_data, self.train_cfg)

        if not self.model_return_loss:  # wh: ?
            outputs.update(self._calc_loss(outputs))

        loss, outputs = self.mutli_task_uncertainty_weighted_loss(outputs)

        if self.model_precision == 'fp16':
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return outputs, loss

    def train_log_visualization(self, outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index):
        losses.update(loss.item())
        losses_dict.update(outputs['loss_dict'])
        if self.global_count % self.print_freq == 0:
            message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(
                epoch, iter_index + 1, len(self.loader), losses_dict.avg(),  # Acc {3} | accuracies.avg(),
                data_time=data_time, run_time=run_time, loss=losses, lr=self.optimizer.param_groups[0]['lr'])
            print(message)
            write2log(self.log_file, '%s\n' % message)
            self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
            self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)

            losses.reset();
            losses_dict.reset();
            data_time.reset()  # accuracies.reset();
            self.summary_writer.flush()

        visual_flag=True
        if visual_flag:
            if self.global_count % (20 * self.print_freq) == 0 or self.global_count == 50:
                vis_ids, vis_errors = determ_worst_best(outputs['kp_error'], top_n=3)

                # vis_ids=[1,2,3,4] # check all sample
                save_name = '{}'.format(self.global_count)
                for ds_name in set(outputs['meta_data']['data_set']):
                    save_name += '_{}'.format(ds_name)
                train_vis_dict, _ = self.visualizer.visulize_result(outputs, outputs['meta_data'],
                                                                    show_items=['mesh','org_img', 'pj2d', 'centermap', 'gt_check', 'foot2d'], \
                                                                    vis_cfg={'settings': ['save_img'], 'vids': vis_ids,
                                                                            'save_dir': self.result_img_dir,
                                                                            'save_name': save_name,
                                                                            'verrors': [vis_errors], 'error_names': ['E']}, global_count=self.global_count)

    def train_epoch(self, epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict = AverageMeter_Dict()
        batch_start_time = time.time()
        t_last=time.time()

        if args().new_training_epoch==epoch:
            # if self.global_count == args().new_training_iters:
            self.train_cfg['new_training'], self.val_cfg['new_training'], self.eval_cfg[
                'new_training'] = False, False, False

        for iter_index, meta_data in enumerate(self.loader):
            self.global_count += 1

            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()

            outputs, loss = self.train_step(meta_data)

            if self.local_rank in [-1, 0]:
                run_time.update(time.time() - run_start_time)
                self.train_log_visualization(outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index)

            if self.global_count % self.test_interval == 0 or self.global_count == self.fast_eval_iter:  # self.print_freq*2
                save_model(self.model, '{}_val_cache.pkl'.format(self.tab), parent_folder=self.model_save_dir)
                # self.validation(epoch)

            if self.distributed_training:
                # wait for rank 0 process finish the job
                torch.distributed.barrier()
            batch_start_time = time.time()

        title = '{}_epoch_{}.pkl'.format(self.tab, epoch)
        if args().test_train:
            if (epoch+1) % 10 ==0: # 10
                save_model(self.model, title, parent_folder=self.model_save_dir)
        else:
            save_model(self.model, title, parent_folder=self.model_save_dir)
        self.e_sche.step()

    # def validation(self, epoch):
    #     logging.info('evaluation result on {} iters: '.format(epoch))
    #     for ds_name, val_loader in self.dataset_val_list.items():
    #         logging.info('Evaluation on {}'.format(ds_name))
    #         MPJPE, PA_MPJPE, eval_results = val_result(self, loader_val=val_loader, evaluation=False)
    #         if ds_name == 'pw3d' and PA_MPJPE < self.val_best_PAMPJPE['pw3d']:
    #             self.val_best_PAMPJPE['pw3d'] = PA_MPJPE
    #             _, _, eval_results = val_result(self, loader_val=self.dataset_test_list['pw3d'], evaluation=True)
    #             self.summary_writer.add_scalars('pw3d-vibe-test', eval_results, self.global_count)
    #         if ds_name == 'mpiinf':
    #             _, _, eval_results = val_result(self, loader_val=self.dataset_test_list['mpiinf'], evaluation=True)
    #             self.summary_writer.add_scalars('mpiinf-test', eval_results, self.global_count)

    #         self.evaluation_results_dict[ds_name]['MPJPE'].append(MPJPE)
    #         self.evaluation_results_dict[ds_name]['PAMPJPE'].append(PA_MPJPE)

    #         logging.info('Running evaluation results:')
    #         ds_running_results = self.get_running_results(ds_name)
    #         print('Running MPJPE:{}|{}; Running PAMPJPE:{}|{}'.format(*ds_running_results))

    #     title = '{}_{:.4f}_{:.4f}_{}.pkl'.format(epoch, MPJPE, PA_MPJPE, self.tab)
    #     logging.info('Model saved as {}'.format(title))
    #     save_model(self.model, title, parent_folder=self.model_save_dir)

    #     self.model.train()
    #     self.summary_writer.flush()

    def get_running_results(self, ds):
        mpjpe = np.array(self.evaluation_results_dict[ds]['MPJPE'])
        pampjpe = np.array(self.evaluation_results_dict[ds]['PAMPJPE'])
        mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var = np.mean(mpjpe), np.var(mpjpe), np.mean(pampjpe), np.var(
            pampjpe)
        return mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var


def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Trainer()
        trainer.train()


if __name__ == '__main__':
    main()
