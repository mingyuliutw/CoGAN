import sys
import itertools
import logging
import torch
import torchvision
import torch.nn as nn
from dataset_mnist import *
from dataset_usps import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from trainer_cogan_mnist2usps import *
from net_config import *
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps/mnist2usps_full_cogan.yaml")


def main(argv):
    (opts, args) = parser.parse_args(argv)
    assert isinstance(opts, object)
    config = NetConfig(opts.config)
    print(config)
    if os.path.exists(config.log):
        os.remove(config.log)
    base_folder_name = os.path.dirname(config.log)
    if not os.path.isdir(base_folder_name):
        os.mkdir(base_folder_name)
    logging.basicConfig(filename=config.log, level=logging.INFO, mode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("Let the journey begin!")
    logging.info(config)
    exec("train_dataset_a = %s(root=config.train_data_a_path, \
                                  num_training_samples=config.train_data_a_size, \
                                  train=config.train_data_a_use_train_data, \
                                  transform=transforms.ToTensor(), \
                                  seed=config.train_data_a_seed)" % config.train_data_a)
    train_loader_a = torch.utils.data.DataLoader(dataset=train_dataset_a, batch_size=config.batch_size, shuffle=True)

    exec("train_dataset_b = %s(root=config.train_data_b_path, \
                                 num_training_samples=config.train_data_b_size, \
                                 train=config.train_data_b_use_train_data, \
                                 transform=transforms.ToTensor(), \
                                seed=config.train_data_b_seed)" % config.train_data_b)
    train_loader_b = torch.utils.data.DataLoader(dataset=train_dataset_b, batch_size=config.batch_size, shuffle=True)

    exec("test_dataset_b = %s(root=config.test_data_b_path, \
                                num_training_samples=config.test_data_b_size, \
                                train=config.test_data_b_use_train_data, \
                                transform=transforms.ToTensor(), \
                                seed=config.test_data_b_seed)" % config.test_data_b)
    test_loader_b = torch.utils.data.DataLoader(dataset=test_dataset_b, batch_size=config.test_batch_size, shuffle=True)

    trainer = MNIST2USPSCoGanTrainer(config.batch_size, config.latent_dims)
    trainer.cuda()
    # Training the Model
    iterations = 0
    while iterations < config.max_iter:
        for it, ((images_a, labels_a), (images_b, labels_b)) in enumerate(
                itertools.izip(train_loader_a, train_loader_b)):
            if images_a.size(0) != config.batch_size or images_b.size(0) != config.batch_size:
                continue
            images_a = Variable(images_a.cuda())
            labels_a = Variable(labels_a.cuda()).view(config.batch_size)
            images_b = Variable(images_b.cuda())
            noise = Variable(torch.randn(config.batch_size, config.latent_dims)).cuda()
            ad_acc, mse_loss, cls_acc = trainer.dis_update(images_a, labels_a, images_b, noise, config.mse_weight,
                                                           config.cls_weight)
            noise = Variable(torch.randn(config.batch_size, config.latent_dims)).cuda()
            fake_images_a, fake_images_b = trainer.gen_update(noise)
            if iterations % config.display == 0 and iterations > 0:
                logging.info("Iteration: %8d, ad_acc: %8.4f, cls_acc: %8.4f, mse_loss: %8.4f" %
                      (iterations, ad_acc, cls_acc, mse_loss.cpu().data.numpy()[0]))
            if iterations % config.snapshot_iter == 0 and iterations > 0:
                # test_score_a = compute_test_score(trainer.dis.classify_a, train_loader_a)
                test_score_b = compute_test_score(trainer.dis.classify_b, test_loader_b)
                logging.info("Classifying Test Dataset B with the cross-domain classifier, acc: %8.4f" % test_score_b)
                dirname = os.path.dirname(config.snapshot_prefix)
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                img_filename = '%s_gen_%08d.jpg' % (config.snapshot_prefix, iterations)
                fake_images = torch.cat((fake_images_a, fake_images_b), 3)
                torchvision.utils.save_image(config.scale*(fake_images.data-config.bias), img_filename)
                # gen_filename = '%s_gen_%08d.pkl' % (config.snapshot_prefix, iterations)
                # dis_filename = '%s_dis_%08d.pkl' % (config.snapshot_prefix, iterations)
                # print("Save generator to %s" % gen_filename)
                # print("Save discriminator to %s" % dis_filename)
                # torch.save(trainer.gen.state_dict(), gen_filename)
                # torch.save(trainer.dis.state_dict(), dis_filename)
            if iterations >= config.max_iter:
                break
            iterations += 1


def compute_test_score(classifier, test_loader):
    score = 0
    num_samples = 0
    for tit, (test_images_b, test_labels_b) in enumerate(test_loader):
        test_images_b = Variable(test_images_b.cuda())
        test_labels_b = Variable(test_labels_b.cuda()).view(test_images_b.size(0))
        cls_outputs = classifier(test_images_b)
        _, cls_predicts = torch.max(cls_outputs.data, 1)
        cls_acc = (cls_predicts == test_labels_b.data).sum()
        score += cls_acc
        num_samples += test_images_b.size(0)
    score /= 1.0 * num_samples
    return score


if __name__ == '__main__':
    main(sys.argv)

