import sys
import logging
import torch
import torchvision
import torch.nn as nn
from dataset_mnistedge import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from trainer_cogan_mnistedge import *
from net_config import *
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps/mnistedge_cogan.yaml")


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

    trainer = MNISTEdgeCoGanTrainer(config.batch_size, config.latent_dims)
    train_dataset = MNISTEDGE(root='../data/mnistedge',
                                train=True,
                                transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    trainer.cuda()
    # Training the Model
    iterations = 0
    while iterations < config.max_iter:
        for it, (images_a, images_b) in enumerate(train_loader):
            if images_a.size(0) != config.batch_size or images_b.size(0) != config.batch_size:
                continue
            images_a = Variable(images_a.cuda())
            images_b = Variable(images_b.cuda())
            noise = Variable(torch.randn(config.batch_size, config.latent_dims)).cuda()
            accuracy = trainer.dis_update(images_a, images_b, noise)
            noise = Variable(torch.randn(config.batch_size, config.latent_dims)).cuda()
            fake_images_a, fake_images_b = trainer.gen_update(noise)
            if iterations % config.display == 0 and iterations > 0:
                logging.info("Iteration: %8d, accuracy: %f" % (iterations, accuracy))
            if iterations % config.snapshot_iter == 0 and iterations > 0:
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


if __name__ == '__main__':
    main(sys.argv)

