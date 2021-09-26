import os.path as osp
import numpy
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
import argparse
from torch.distributions import normal, kl
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder, ARGVA
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from losses import*
from model import*
from preprocess import*
from centrality import *
warnings.filterwarnings("ignore")
#  GAN
aligner = Aligner()
generator1 = Generator1()
discriminator1 = Discriminator1()
generator2 = Generator2()
discriminator2 = Discriminator2()
# Losses
adversarial_loss1 = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# send 1st GAN to GPU
aligner.to(device)
generator1.to(device)
discriminator1.to(device)
generator2.to(device)
discriminator2.to(device)
adversarial_loss1.to(device)
l1_loss.to(device)

Aligner_optimizer = torch.optim.AdamW(aligner.parameters(), lr=0.025, betas=(0.5, 0.999))
generator1_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.025, betas=(0.5, 0.999))
discriminator1_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))
generator2_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.025, betas=(0.5, 0.999))
discriminator2_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))

def StairwayGraphNet (X_train_source, X_test_source, X_train_target, X_test_target):

    X_casted_train_source = cast_data_vector_RH(X_train_source)
    X_casted_test_source = cast_data_vector_RH(X_test_source)
    X_casted_train_target1 = cast_data_vector_FC(X_train_target1)
    X_casted_test_target1 = cast_data_vector_FC(X_test_target1)
    X_casted_train_target2 = cast_data_vector_HHR(X_train_target2)
    X_casted_test_target2 = cast_data_vector_HHR(X_test_target2)

    aligner.train()
    generator1.train()
    discriminator1.train()
    generator2.train()
    discriminator2.train()


    nbre_epochs = 2
    for epochs in range(nbre_epochs):
        # Train Generator
        with torch.autograd.set_detect_anomaly(True):
            Al_losses = []


            Ge_losses1 = []
            losses_discriminator1= []
            Ge_losses2 = []
            losses_discriminator2= []

            i = 0
            for data_source, data_target1, data_target2 in zip(X_casted_train_source, X_casted_train_target1, X_casted_train_target2):
                # print(i)
                targett = data_target.edge_attr.view(160, 160)
                targett2 = data_target.edge_attr.view(268, 268)
                # ************    Domain alignment    ************
                A_output = aligner(data_source)
                A_casted = convert_generated_to_graph_Al(A_output)
                A_casted = A_casted[0]

                target = data_target.edge_attr.view(160, 160).detach().cpu().clone().numpy()
                target_mean = np.mean(target)
                target_std = np.std(target)

                d_target = torch.normal(target_mean, target_std, size=(1, 595))
                dd_target = cast_data_vector_RH(d_target)
                dd_target = dd_target[0]
                target_d = dd_target.edge_attr.view(35, 35)

                kl_loss = Alignment_loss(target_d, A_output)

                Al_losses.append(kl_loss)

                # ************     Super-resolution 1   ************
                G_output = generator(A_casted)  # 35 x 35
                # print("G_output: ", G_output.shape)
                G_output_reshaped = (G_output.view(1, 160, 160, 1).type(torch.FloatTensor)).detach()
                G_output_casted = convert_generated_to_graph(G_output_reshaped)
                G_output_casted = G_output_casted[0]
                torch.cuda.empty_cache()

                Gg_loss = GT_loss(targett, G_output)
                torch.cuda.empty_cache()
                D_real = discriminator1(data_target)
                D_fake = discriminator1(G_output_casted)
                torch.cuda.empty_cache()
                G_adversarial = adversarial_loss(D_fake, (torch.ones_like(D_fake, requires_grad=False)))
                G_loss = G_adversarial + Gg_loss
                Ge_losses1.append(G_loss)

                D_real_loss = adversarial_loss(D_real, (torch.ones_like(D_real, requires_grad=False)))
                # torch.cuda.empty_cache()
                D_fake_loss = adversarial_loss(D_fake.detach(), torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                # torch.cuda.empty_cache()
                losses_discriminator.append(D_loss)
                i += 1
                # ************     Super-resolution 2   ************
                G_output2 = generator(G_output)  # 35 x 35
                # print("G_output: ", G_output.shape)
                G_output_reshaped2 = (G_output2.view(1, 268, 268, 1).type(torch.FloatTensor)).detach()
                G_output_casted2 = convert_generated_to_graph_HHR(G_output_reshaped2)
                G_output_casted2 = G_output_casted2[0]
                torch.cuda.empty_cache()

                Gg_loss2 = GT_loss(targett2, G_output2)
                torch.cuda.empty_cache()
                D_real2 = discriminator2(data_target2)
                D_fake2 = discriminator2(G_output_casted2)
                torch.cuda.empty_cache()
                G_adversarial2 = adversarial_loss(D_fake2, (torch.ones_like(D_fake2, requires_grad=False)))
                G_loss2 = G_adversarial2 + Gg_loss2
                Ge_losses2.append(G_loss2)

                D_real_loss2 = adversarial_loss(D_real2, (torch.ones_like(D_real2, requires_grad=False)))
                # torch.cuda.empty_cache()
                D_fake_loss2 = adversarial_loss(D_fake2.detach(), torch.zeros_like(D_fake2))
                D_loss2 = (D_real_loss2 + D_fake_loss2) / 2
                # torch.cuda.empty_cache()
                losses_discriminator2.append(D_loss2)
                i += 1

            # torch.cuda.empty_cache()
            generator2_optimizer.zero_grad()
            Ge_losses2 = torch.mean(torch.stack(Ge_losses2))
            Ge_losses2.backward(retain_graph=True)
            generator2_optimizer.step()

            generator_optimizer.zero_grad()
            Ge_losses = torch.mean(torch.stack(Ge_losses))
            Ge_losses.backward(retain_graph=True)
            generator_optimizer.step()

            Aligner_optimizer.zero_grad()
            Al_losses = torch.mean(torch.stack(Al_losses))
            Al_losses.backward(retain_graph=True)
            Aligner_optimizer.step()

            discriminator2_optimizer.zero_grad()
            losses_discriminator2 = torch.mean(torch.stack(losses_discriminator2))
            losses_discriminator2.backward(retain_graph=True)
            discriminator2_optimizer.step()

            discriminator_optimizer.zero_grad()
            losses_discriminator = torch.mean(torch.stack(losses_discriminator))
            losses_discriminator.backward(retain_graph=True)
            discriminator_optimizer.step()

        print("[Epoch: %d]| [Al loss: %f]| [Ge1 loss: %f]| [D1 loss: %f] [Ge2 loss: %f]| [D2 loss: %f]" % (epochs, Al_losses, Ge_losses, losses_discriminator, Ge_losses2, losses_discriminator2))

    torch.save(aligner.state_dict(), "./weight" + "aligner_fold" + "_" + ".model")
    torch.save(generator1.state_dict(), "./weight" + "generator_fold" + "_" + ".model")
    torch.save(generator2.state_dict(), "./weight" + "generator_fold" + "_" + ".model")

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    # #     ######################################### TESTING PART #########################################
    restore_aligner = "./weight" + "aligner_fold" + "_" + ".model"
    restore_generator1 = "./weight" + "generator1_fold" + "_" + ".model"
    restore_generator2 = "./weight" + "generator2_fold" + "_" + ".model"

    aligner.load_state_dict(torch.load(restore_aligner))
    generator2.load_state_dict(torch.load(restore_generator2))
    generator1.load_state_dict(torch.load(restore_generator1))

    aligner.eval()
    generator1.eval()
    generator2.eval()

    i = 0
    predicted_test_graphs1 = []
    losses_test1 = []
    eigenvector_losses_test1 = []
    l1_tests1 = []
    Closeness_test1 = []
    Eigenvector_test1 = []

    predicted_test_graphs2 = []
    losses_test2 = []
    eigenvector_losses_test2 = []
    l1_tests2 = []
    Closeness_test2 = []
    Eigenvector_test2 = []
    for data_source, data_target1, data_target2 in zip(X_casted_test_source, X_casted_test_target1, X_casted_test_target2):
        # print(i)
        data_source_test = data_source.x.view(35, 35)
        data_target_test1 = data_target1.x.view(160, 160)
        data_target_test2 = data_target2.x.view(268, 268)


        A_test = aligner(data_source)
        A_test_casted = convert_generated_to_graph_Al(A_test)
        A_test_casted = A_test_casted[0]
        data_target1 = data_target_test1.detach().cpu().clone().numpy()
        # ************     Super-resolution 1   ************
        G_output_test = generator(A_test_casted)  # 35 x35
        G_output_test_casted = convert_generated_to_graph(G_output_test)
        G_output_test_casted = G_output_test_casted[0]
        torch.cuda.empty_cache()

        L1_test1 = l1_loss(data_target_test1, G_output_test)
        # fold= 1
        target_test = data_target_test1.detach().cpu().clone().numpy()
        predicted_test = G_output_test.detach().cpu().clone().numpy()
        source_test = data_source_test.detach().cpu().clone().numpy()

        torch.cuda.empty_cache()
        fake_topology_test = torch.tensor(topological_measures(predicted_test))
        real_topology_test = torch.tensor(topological_measures(target_test))

        eigenvector_test1 = (l1_loss(fake_topology_test[2], real_topology_test[2]))


        l1_tests1.append(L1_test1.detach().cpu().numpy())
        Eigenvector_test1.append(eigenvector_test1.detach().cpu().numpy())
        # ************     Super-resolution 2   ************
        G_output_test2 = generator(G_output_test)  # 35 x35
        G_output_test_casted2 = convert_generated_to_graph_HHR(G_output_test2)
        G_output_test_casted2 = G_output_test_casted2[0]
        torch.cuda.empty_cache()

        L1_test2 = l1_loss(data_target_test, G_output_test)
        # fold= 1
        target_test2 = data_target_test2.detach().cpu().clone().numpy()
        predicted_test2 = G_output_test2.detach().cpu().clone().numpy()


        torch.cuda.empty_cache()
        fake_topology_test2 = torch.tensor(topological_measures(predicted_test2))
        real_topology_test2 = torch.tensor(topological_measures(target_test2))

        eigenvector_test2 = (l1_loss(fake_topology_test2[2], real_topology_test2[2]))


        l1_tests2.append(L1_test2.detach().cpu().numpy())
        Eigenvector_test2.append(eigenvector_test2.detach().cpu().numpy())



    mean_l11 = np.mean(l1_tests1)
    mean_eigenvector2 = np.mean(Eigenvector_test1)
    mean_l12 = np.mean(l1_tests2)
    mean_eigenvector2 = np.mean(Eigenvector_test2)

    # print("Mean L1 Loss Test: ", fold_mean_l1_loss)
    # print()

    losses_test1.append(mean_l11)
    eigenvector_losses_test1.append(mean_eigenvector1)

    losses_test2.append(mean_l12)
    eigenvector_losses_test2.append(mean_eigenvector2)

    # fold += 1
    return (source_test, predicted_test1, data_target1, losses_test1, eigenvector_losses_test1, predicted_test2, data_target2, losses_test2, eigenvector_losses_test2)



