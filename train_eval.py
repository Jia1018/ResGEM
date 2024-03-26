import time
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from Dataloader import get_random_rm

def adjust_learning_rate(optimizer, epoch, opt):
    lr_shceduler(optimizer, epoch, opt.lr)

def lr_shceduler(optimizer, epoch, init_lr):
    if epoch > 36:
        init_lr *= 0.5e-3
    elif epoch > 32:
        init_lr *= 1e-3
    elif epoch > 24:
        init_lr *= 1e-2
    elif epoch > 16:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


def print_info(info, logfile):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, '
               'Train Loss: {:.4f}, Train MSE: {:.4f}, Test Loss:{:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss'], info['MSE'], info['test_loss'])
    print(message, file=logfile)
    print(message)


def run(args, model, train_loader, test_loader, epochs, optimizer,
        scheduler, device, logfile):

    min_loss = 999.
    min_epoch = 0
    for epoch in range(args.start_epoch, epochs + 1):
        t = time.time()
        #adjust_learning_rate(optimizer, epoch, args)
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']), file=logfile)
        train_loss, MSE = train(args, model, train_loader, optimizer, device, epoch)
        #train_loss = new_train(args, model, train_loader, optimizer, device, epoch, logfile)
        t_duration = time.time() - t
        scheduler.step()
        test_loss = test(args, model, test_loader, device, epoch)
        #test_loss = new_test(args, model, test_loader, device, epoch, logfile)
        eval_info = {
            'train_loss': train_loss,
            "MSE": MSE, 
            'test_loss': test_loss,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration
        }

        if(test_loss < min_loss):
            min_loss = test_loss
            min_epoch = epoch
            checpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(checpoint_state, '%s/model_full_best.pth' % args.network_model_dir)
            

        print_info(eval_info, logfile)
    
    print("min loss: %f || min epoch %d" % (min_loss, min_epoch), file=logfile)


def cal_normal_dot_loss(pred_results, gt_vertex_positions, gt_face_normals, face_vertex_indices, edge_index, alpha):
    pts_0 = pred_results[face_vertex_indices.T[0]]
    pts_1 = pred_results[face_vertex_indices.T[1]]
    pts_2 = pred_results[face_vertex_indices.T[2]]
    edges_0 = F.normalize(pts_1 - pts_0)
    edges_1 = F.normalize(pts_2 - pts_1)
    edges_2 = F.normalize(pts_0 - pts_2)
    dot1 = torch.square(torch.mul(gt_face_normals, edges_0).sum(1))
    dot2 = torch.square(torch.mul(gt_face_normals, edges_1).sum(1))
    dot3 = torch.square(torch.mul(gt_face_normals, edges_2).sum(1))
    mean_dot_error = (dot1.sum() + dot2.sum() + dot3.sum())/(gt_face_normals.shape[0] * 3)
    gt_mean_edge_length = torch.sqrt(torch.square(gt_vertex_positions[edge_index[0]] - gt_vertex_positions[edge_index[1]]).sum(1)).mean()
    mean_edge_length = torch.sqrt(torch.square(pred_results[edge_index[0]] - pred_results[edge_index[1]]).sum(1)).mean()

    loss = alpha * mean_dot_error + (1 - alpha) * (mean_edge_length - gt_mean_edge_length) ** 2
    
    return loss

def cal_normal_loss(pred_results, gt_vertex_positions, gt_face_normals, face_vertex_indices):
    pts_0 = pred_results[face_vertex_indices.T[0]]
    pts_1 = pred_results[face_vertex_indices.T[1]]
    pts_2 = pred_results[face_vertex_indices.T[2]]
    edges_1 = (pts_1 - pts_0).T
    edges_2 = (pts_2 - pts_0).T
    pred_face_normals = torch.cat([(torch.mul(edges_1[1], edges_2[2]) - torch.mul(edges_2[1], edges_1[2])).unsqueeze(1),
                                    (torch.mul(edges_1[2], edges_2[0]) - torch.mul(edges_2[2], edges_1[0])).unsqueeze(1),
                                    (torch.mul(edges_1[0], edges_2[1]) - torch.mul(edges_2[0], edges_1[1])).unsqueeze(1)], dim=1)
    pred_face_normals = F.normalize(pred_face_normals)
    #normal_loss = torch.square(1 - torch.abs(torch.mul(gt_face_normals, pred_face_normals).sum(1))).mean()
    normal_loss = F.mse_loss(gt_face_normals, pred_face_normals)
    return normal_loss


def train(args, model, train_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    total_MSE = 0
    for idx, data in enumerate(train_loader):
        
        for d in data:
            Rotmat = get_random_rm()
            d.x[:, :3] = torch.mm(d.x[:, :3], Rotmat)
            d.x[:, 3:6] = torch.mm(d.x[:, 3:6], Rotmat)
            if args.graph_type == 'VERTEX':
                d.gt_vertex_positions = torch.mm(d.gt_vertex_positions, Rotmat)
                d.gt_face_normals = torch.mm(d.gt_face_normals.T, Rotmat).T
            else:
                d.gt_face_normals = torch.mm(d.gt_face_normals, Rotmat)
            d.rotate_matrix = torch.mm(d.rotate_matrix, Rotmat)
            if args.graph_type =='DENSE_FACE':
                d.edge_index = d.dense_edge_index
                d.edge_attr = d.dense_edge_attr
        
        #data = data.to(device)
        optimizer.zero_grad()
        pred_results = model(data)
        if args.graph_type =='VERTEX':
            target = [d.gt_vertex_positions for d in data]
            target = torch.cat(target).to(device)
            pred_results += torch.cat([d.x[:, :3] for d in data])[:, :3].to(device)
            
            face_vertex_indices = [d.face_vertex_indices.T + i * args.max_patch_size for i, d in enumerate(data)]
            face_vertex_indices = torch.cat(face_vertex_indices, dim=0).to(device)
            gt_face_normals = [d.gt_face_normals.T for d in data]
            gt_face_normals = torch.cat(gt_face_normals, dim=0).to(device)
            loss2 = 100 * cal_normal_loss(pred_results, target, gt_face_normals, face_vertex_indices)
            
        else:
            pred_results = F.normalize(pred_results)
            target = [d.gt_face_normals for d in data]
            target = torch.cat(target).to(device)
            edge_index = [d.edge_index + i * args.max_patch_size for i, d in enumerate(data)]
            edge_index = torch.cat(edge_index, dim=1).to(device)
            loss2 = F.mse_loss(pred_results[edge_index[0]], pred_results[edge_index[1]])

        MSE = 100 * F.mse_loss(pred_results, target)
        
        loss = MSE + loss2
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f mse: %f' % (epoch, idx, len(train_loader), loss.item(), MSE*0.01))
        total_loss += loss.item()
        total_MSE += MSE.item()

    checpoint_state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}

    if epoch == (args.nepoch - 1):
        torch.save(checpoint_state, '%s/model_full_ae.pth' % args.network_model_dir)

    if epoch % args.model_interval == 0:
        torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (args.network_model_dir, epoch))

    return total_loss / len(train_loader), total_MSE / len(train_loader)


def test(args, model, test_loader, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            for d in data:
                Rotmat = get_random_rm()
                d.x[:, :3] = torch.mm(d.x[:, :3], Rotmat)
                d.x[:, 3:6] = torch.mm(d.x[:, 3:6], Rotmat)
                if args.graph_type == 'VERTEX':
                    d.gt_vertex_positions = torch.mm(d.gt_vertex_positions, Rotmat)
                    d.gt_face_normals = torch.mm(d.gt_face_normals.T, Rotmat).T
                else:
                    d.gt_face_normals = torch.mm(d.gt_face_normals, Rotmat)
                d.rotate_matrix = torch.mm(d.rotate_matrix, Rotmat)
                if args.graph_type =='DENSE_FACE':
                    d.edge_index = d.dense_edge_index
                    d.edge_attr = d.dense_edge_attr
            pred_results = model(data)
            if args.graph_type =='VERTEX':
                target = [d.gt_vertex_positions for d in data]
                target = torch.cat(target).to(device)
                pred_results += torch.cat([d.x[:, :3] for d in data])[:, :3].to(device)
            else:
                pred_results = F.normalize(pred_results)
                target = [d.gt_face_normals for d in data]
                target = torch.cat(target).to(device)
                
            loss1 = 100 * F.mse_loss(pred_results, target)
            loss = loss1
            total_loss += loss
    return total_loss / len(test_loader)

def infer_full_model(args, model, model_name, infer_loader, result_folder_name, min_patch_size):
    data_folder = os.path.join(args.testset, args.data_type, model_name)
    if args.graph_type == 'VERTEX':
        node_num = (np.loadtxt(os.path.join(data_folder, "gt_vertex_positions.txt"))).shape[0]
    else:
        node_num = (np.loadtxt(os.path.join(data_folder, "gt_face_normals.txt"))).shape[0]
    #node_num = gt_face_normals.shape[0]
    model.eval()
    with torch.no_grad():
        full_pred_results = np.zeros((node_num, 3))
        repeat_cnt = np.zeros(node_num)
        for data in infer_loader:
            #print(data[0].name)
            #data = data.to(device)
            if args.graph_type =='DENSE_FACE':
                data[0].edge_index = data[0].dense_edge_index
                data[0].edge_attr = data[0].dense_edge_attr
            if data[0].x.shape[0] >= min_patch_size:
                pred_results = model(data).data.cpu()
            else:
                if args.graph_type =='VERTEX':
                    pred_results = torch.zeros_like(data[0].x[:, :3])
                else:
                    pred_results = data[0].x[:, :3]

            if args.graph_type =='VERTEX':
                pred_results = pred_results + data[0].x[:, :3]
                #data.x[:, :3] = pred_results
                pred_results = torch.mm(pred_results, torch.inverse(data[0].rotate_matrix))
                pred_results = pred_results * data[0].avg_edge_length
                pred_results = pred_results + data[0].center
                pred_results = pred_results.numpy()
            else:
                pred_results = F.normalize(pred_results)
                pred_results = torch.mm(pred_results, torch.inverse(data[0].rotate_matrix))
                pred_results = pred_results.numpy()
            
            if data[0].x.shape[0] == 1:
                pred_results = pred_results.squeeze(0)
            full_pred_results[data[0].nodesInd] += pred_results
            repeat_cnt[data[0].nodesInd] += 1
            
        full_pred_results /= np.expand_dims(repeat_cnt, 1).repeat(3, axis=1)
        if args.graph_type == 'VERTEX':
            save_folder = os.path.join(args.save_dir, args.data_type, 'V', result_folder_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, model_name + '_V.txt'), 
                                full_pred_results.astype('float32'), fmt='%f')
        else:
            save_folder = os.path.join(args.save_dir, args.data_type, 'NN', result_folder_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, model_name + '_NN.txt'), 
                                full_pred_results.astype('float32'), fmt='%f')
            