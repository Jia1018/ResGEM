import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
#from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F

import time
import os
import shutil
import numpy as np
import random
from random import sample
from math import sin, cos, pi
import scipy.spatial as sp

try:
    import queue
except ImportError:
    import Queue as queue

def creatAdj(edge_index, node_num, K):
    AdjOut = np.zeros((node_num, K),dtype=int) - 1
    adj_dict = {}
    for pair in edge_index:
        node_0 = pair[0]
        node_1 = pair[1]
        if node_0 not in adj_dict:
            adj_dict[node_0] = [node_0, node_1]
        else:
            adj_dict[node_0].append(node_1)

    for node_id in adj_dict:
        for i in range(min(len(adj_dict[node_id]), K)):
            AdjOut[int(node_id)][i] = adj_dict[node_id][i]

    return AdjOut

def getGraphPatch_wMask(Adj, max_patch_size, seed, mask, min_patch_size):

    K = Adj.shape[1]
    AdjOut = np.zeros((max_patch_size + K, K),dtype=int)-1

    nodesNewInd = np.zeros(Adj.shape[0],dtype=int)  # Array of correspondence between old nodes indices and new ones
    nodesNewInd -= 1
    nodesOldInd = np.zeros(Adj.shape[0],dtype=int)  # Array of correspondence between new nodes indices and old ones
    nodesOldInd -= 1

    nIt = [0]   # Using list as a dirty trick for namespace reasons for inner functions

    #Adj = Adj-1   # Switch to zero-indexing

    # Update correspondance tables
    def addNode(nind):
        nodesNewInd[nind] = nIt[0]
        nodesOldInd[nIt[0]] = nind
        nIt[0] += 1

    nQueue = queue.Queue()          # Queue of nodes to be added
    nQueue.put(seed)                # Add seed to start the process
    borderQueue = queue.Queue()

    #print("fQueue: "+str(fQueue.empty()))

    addNode(seed)

    while (nIt[0]<max_patch_size):    # Keep growing until we reach desired count
        if nQueue.empty():
            break

        curN = nQueue.get()             # Get current node index
        newNInd = nodesNewInd[curN]     # Get its new index
        # current node should already be added

        AdjOut[newNInd,0] = newNInd  #Add itself first.

        # Process neighbours
        for neigh in range(1,K):    # Skip first entry, its the current face
            
            curNei = Adj[curN,neigh]

            if curNei==-1:    # We've reached the last neighbour
                break

            if(nodesNewInd[curNei]==-1):  # If face not processed, add it to faces list and add it to queue
                addNode(curNei)
                if (nIt[0]>=max_patch_size):
                    break
                if mask[curNei]==1:
                    borderQueue.put(curNei)
                else:
                    nQueue.put(curNei)

            AdjOut[newNInd,neigh] = nodesNewInd[curNei]    #fill new adj graph

    nextSeed = -1

    # We've reached the count OR filled that region of the graph.

    # The following code makes sure patches have a minimum size 
    if nIt[0]<min_patch_size:
        # In this case, ignore mask and just keep growing until the desired size is reached.
        # First, empty border queue, and keep going normally with standard nQueue.
        #print("Local region complete. Keep growing patch for context")
        #print("(current patch size = %i)"%nIt[0])

        while (nIt[0]<min_patch_size):    # Keep growing until we reach desired count
            if borderQueue.empty():
                break

            curN = borderQueue.get()        # Get current node index
            newNInd = nodesNewInd[curN]     # Get its new index
            # current node should already be added

            AdjOut[newNInd,0] = newNInd  #Add itself first.

            # Process neighbours
            for neigh in range(1,K):    # Skip first entry, its the current face
                
                curNei = Adj[curN,neigh]

                if curNei==-1:    # We've reached the last neighbour
                    break

                if(nodesNewInd[curNei]==-1):  # If face not processed, add it to faces list and add it to queue
                    addNode(curNei)
                    if (nIt[0]>=min_patch_size):
                        break
                    nQueue.put(curNei)

                AdjOut[newNInd,neigh] = nodesNewInd[curNei]    #fill new adj graph
        
        while (nIt[0]<min_patch_size):    # Keep growing until we reach desired count
            if nQueue.empty():
                break

            curN = nQueue.get()             # Get current node index
            newNInd = nodesNewInd[curN]     # Get its new index
            # current node should already be added


            AdjOut[newNInd,0] = newNInd  #Add itself first.

            # Process neighbours
            for neigh in range(1,K):    # Skip first entry, its the current face
                
                curNei = Adj[curN,neigh]

                if curNei==-1:    # We've reached the last neighbour
                    break

                if(nodesNewInd[curNei]==-1):  # If face not processed, add it to faces list and add it to queue
                    addNode(curNei)
                    if (nIt[0]>=min_patch_size):
                        break
                    nQueue.put(curNei)

                AdjOut[newNInd,neigh] = nodesNewInd[curNei]    #fill new adj graph

    # Now, the patch has reached the desired size either way.
    # We just need to complete adjacency matrix for all border nodes.

    #print("patch filled up. Processing remaining nodes in queue")
    # Now, fill adjacency graph for remaining nodes in the queue
    while not nQueue.empty():
        curN = nQueue.get()         # Get current face index
        
        newNInd = nodesNewInd[curN]     # Get its new index

        AdjOut[newNInd,0] = newNInd  #Add itself first.

        neighCount = 1
        # Process neighbours
        for neigh in range(1,K):    # Skip first entry, its the current face
            
            curNei = Adj[curN,neigh]

            if curNei==-1:    # We've reached the last neighbour
                break

            if(nodesNewInd[curNei]==-1):  # If face not in the graph, skip it
                if(mask[curNei]==0):
                    nextSeed = curNei 
                continue

            AdjOut[newNInd,neighCount] = nodesNewInd[curNei]    #fill new adj graph
            neighCount += 1


    # Do the same thing with border queue
    while not borderQueue.empty():
        curN = borderQueue.get()         # Get current face index
        
        newNInd = nodesNewInd[curN]     # Get its new index

        AdjOut[newNInd,0] = newNInd  #Add itself first.

        neighCount = 1
        # Process neighbours
        for neigh in range(1,K):    # Skip first entry, its the current face
            
            curNei = Adj[curN,neigh]

            if curNei==-1:    # We've reached the last neighbour
                break

            if(nodesNewInd[curNei]==-1):  # If face not in the graph, skip it
                if(mask[curNei]==0):
                    nextSeed = curNei 
                continue

            AdjOut[newNInd,neighCount] = nodesNewInd[curNei]    #fill new adj graph
            neighCount += 1

    AdjOut = AdjOut[:nIt[0],:]
    nodesOldInd = nodesOldInd[:nIt[0]]
    #print("final patch size = %i"%nIt[0])

    #AdjOut = AdjOut+1     # Switch back to one-indexing

    return AdjOut, nodesOldInd, nodesNewInd, nextSeed, nIt[0]    # return nodesOldInd as well

def get_random_rm():
    theta_x = random.randint(0, 180) * pi / 180
    theta_y = random.randint(0, 180) * pi / 180
    theta_z = random.randint(0, 180) * pi / 180
    R_x = torch.tensor(np.array([[1, 0, 0], [0, cos(theta_x), sin(theta_x)], [0, -sin(theta_x), cos(theta_x)]]), dtype=torch.float)
    R_y = torch.tensor(np.array([[cos(theta_y), 0, -sin(theta_y)], [0, 1, 0], [sin(theta_y), 0, cos(theta_y)]]), dtype=torch.float)
    R_z = torch.tensor(np.array([[cos(theta_z), sin(theta_z), 0], [-sin(theta_z), cos(theta_z), 0], [0, 0, 1]]), dtype=torch.float)
    rotate_matrix = torch.mm(torch.mm(R_x, R_y), R_z)
    return rotate_matrix


class InferDataset_full_model(Dataset):
    def __init__(self, root, parser, model_name='', patch_size=20000, transform=None, pre_transform=None):
        self.data_path = os.path.join(parser.testset, parser.data_type, model_name)
        self.model_name = model_name
        self.data_list = []
        self.patch_size = patch_size
        self.max_neighbor_num = parser.max_neighbor_num
        self.max_one_ring_num = parser.max_one_ring_num
        self.graph_type = parser.graph_type
        self.neighbor_sizes = parser.neighbor_sizes
        super(InferDataset_full_model, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\GraphDataset.dataset']
    
    def download(self):
        pass

    def process(self):
        #return
        patch_folder = self.data_path
        v_features = np.loadtxt(os.path.join(patch_folder, "v_features.txt"))
        f_features = np.loadtxt(os.path.join(patch_folder, "f_features.txt"))
        vertex_edge_index = np.loadtxt(os.path.join(patch_folder, "vertex_edge_index.txt"))
        face_edge_index = np.loadtxt(os.path.join(patch_folder, "face_edge_index.txt"))
        face_edge_weight = np.loadtxt(os.path.join(patch_folder, "face_edge_weight.txt"))
        face_dense_edge_index = np.loadtxt(os.path.join(patch_folder, "face_dense_edge_index.txt"))
        face_dense_edge_weight = np.loadtxt(os.path.join(patch_folder, "face_dense_edge_weight.txt"))

        input_gt_vertex_positions = np.loadtxt(os.path.join(patch_folder, "gt_vertex_positions.txt"))
        avg_edge_length = torch.tensor(np.loadtxt(os.path.join(patch_folder, "avg_edge_length.txt")), dtype=torch.float)
        center = torch.tensor(np.loadtxt(os.path.join(patch_folder, "center.txt")), dtype=torch.float)



        if self.graph_type == 'VERTEX':
            input_feature = v_features[:, :6]
            input_edge_index = vertex_edge_index
            input_edge_weight = np.zeros((input_edge_index.shape[0], 1))
            node_num = input_feature.shape[0]         
            Adj = creatAdj(input_edge_index, node_num, self.max_neighbor_num)
        else: 
            input_feature = f_features[:, :7]
            input_edge_index = face_edge_index
            input_edge_weight = face_edge_weight
            input_dense_edge_index = face_dense_edge_index
            input_dense_edge_weight = face_dense_edge_weight
            node_num = input_feature.shape[0]
            Adj = creatAdj(input_dense_edge_index, node_num, self.max_neighbor_num)

        nodeCheck = np.zeros(node_num)
        nodeRange = np.arange(node_num)

        patchNum = 0
        #continue
        #print("Dividing mesh into patches: %i nodes (%i max allowed)"%(node_num,self.patch_size))
                
        nextSeed = -1
        while(np.any(nodeCheck==0)):
            toBeProcessed = nodeRange[nodeCheck==0]
            if nextSeed==-1:
                nodeSeed = np.random.randint(toBeProcessed.shape[0])
                nodeSeed = toBeProcessed[nodeSeed]
            else:
                nodeSeed = nextSeed
                if nodeCheck[nodeSeed]==1:
                    #print("ERROR: Bad seed returned!!")
                    return
                
            tp0 = time.clock()

            testPatchAdj, nodesInd, nodesInd_mask, nextSeed, patch_node_num = getGraphPatch_wMask(Adj, self.patch_size, nodeSeed, nodeCheck, self.patch_size)
            tp1 = time.clock()
            #print("Mesh patch extracted ("+str(1000*(tp1-tp0))+"ms)")

            nodeCheck[nodesInd]=1
            #print("Total added nodes = %i"%np.sum(nodeCheck==1))
            #print(nodesInd)

            x = input_feature[nodesInd]
            if self.graph_type == 'VERTEX':
                pass
                
                gt_vertex_positions = input_gt_vertex_positions[nodesInd]
                gt_vertex_positions = torch.tensor(gt_vertex_positions, dtype=torch.float)

            else:
                edgeInd = []
                edgeW = []
                for idx, pair in enumerate(input_dense_edge_index):
                    if nodesInd_mask[int(pair[0])] != -1 and nodesInd_mask[int(pair[1])] != -1:
                        edgeInd.append([nodesInd_mask[int(pair[0])], nodesInd_mask[int(pair[1])]])
                        edgeW.append(input_dense_edge_weight[idx])
                dense_edge_index = edgeInd
                dense_edge_weight = edgeW
                dense_edge_index = torch.tensor(dense_edge_index, dtype=torch.long).T  # edge_index should be long type
                dense_edge_weight = torch.tensor(dense_edge_weight, dtype=torch.float) # edge_index should be float type
                        
            edgeInd = []
            edgeW = []
            for idx, pair in enumerate(input_edge_index):
                if nodesInd_mask[int(pair[0])] != -1 and nodesInd_mask[int(pair[1])] != -1:
                    edgeInd.append([nodesInd_mask[int(pair[0])], nodesInd_mask[int(pair[1])]])
                    edgeW.append(input_edge_weight[idx])
            edge_index = edgeInd
            edge_weight = edgeW

            edge_index = torch.tensor(edge_index, dtype=torch.long).T  # edge_index should be long type
            edge_weight = torch.tensor(edge_weight, dtype=torch.float) # edge_index should be float type
            
            nodesInd = torch.tensor(nodesInd, dtype=torch.long)

            x = torch.tensor(x, dtype=torch.float) 
                        
            # y should be long type, graph label should not be a 0-dimesion tensor
            # use [graph_label[i]] ranther than graph_label[i]
            y = torch.tensor([patchNum], dtype=torch.long) 

            if self.graph_type == 'VERTEX':
                node_positions = x[:, :3]
                node_normals = x[:, 3:6]
            else:
                node_positions = x[:, 3:6]
                node_normals = x[:, :3]
                        
            rotate_matrix = get_random_rm()
                        
            node_positions = node_positions - center
            node_positions /= avg_edge_length
            node_positions = torch.mm(node_positions, rotate_matrix)
            node_normals = torch.mm(node_normals, rotate_matrix)
                        
            if self.graph_type == 'VERTEX':
                x[:, :3] = node_positions
                x[:, 3:6] = node_normals
            else:
                x[:, :3] = node_normals
                x[:, 3:6] = node_positions
                x[:, 6:] /= torch.square(avg_edge_length)

            if self.graph_type == 'VERTEX':
                data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight, name=self.model_name+'_'+str(patchNum), 
                        gt_vertex_positions=gt_vertex_positions,
                        avg_edge_length=avg_edge_length, center=center, rotate_matrix=rotate_matrix, nodesInd=nodesInd)
            else:
                data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight, name=self.model_name+'_'+str(patchNum), 
                        dense_edge_index=dense_edge_index, dense_edge_attr=dense_edge_weight,
                        avg_edge_length=avg_edge_length, center=center, rotate_matrix=rotate_matrix, nodesInd=nodesInd)            
            
            self.data_list.append(data)

            patchNum += 1
            
        #print("Patch number for model", self.model_name, ":", patchNum)        

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        return data

