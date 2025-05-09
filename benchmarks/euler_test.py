from copy import deepcopy
import argparse

import pandas as pd
import torch 
from torch.optim import Adam
from torch_geometric.nn import Node2Vec

import generators as g
import loaders.load_vgrnn as vd
from models.euler_serial import EulerGCN
from utils import get_score

import wandb
wandb.login()

torch.set_num_threads(8)

NUM_TESTS = 5
PATIENCE = 100
MAX_DECREASE = 2
TEST_TS = 3

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(run, dataset, model, data, epochs=1500, pred=False, nratio=1, lr=0.01):
    print(lr)
    end_tr = data.T-TEST_TS
    
    if not pred:  
        tr_nodes = data.tr(0)[0]
        for i in range(1,end_tr):
            torch.cat((tr_nodes,data.tr(i)[0]))
        tr_nodes = torch.unique(tr_nodes)
      
        test_nodes = data.tr(end_tr)[0]
        for i in range(end_tr+1,data.T):
            torch.cat((test_nodes,data.tr(i)[0]))
        test_nodes = torch.unique(test_nodes)

        mask_new = torch.zeros(data.x.size(0))
        mask_old = torch.ones(data.x.size(0))
        nb_new_nodes = 0

        for node in test_nodes:
            if node not in tr_nodes:
                nb_new_nodes+=1
            else:
                mask_new[node] = 1
            #print("new_nodes:",nb_new_nodes)
        mask_old = mask_old - mask_new
    else:
        tr_nodes = data.tr(0)[0]
        for i in range(1,end_tr-1):
            torch.cat((tr_nodes,data.tr(i)[0]))
        tr_nodes = torch.unique(tr_nodes)

        test_nodes = data.all(end_tr-1)[0]
        for i in range(end_tr,data.T-1):
            torch.cat((test_nodes,data.all(i)[0]))
        test_nodes = torch.unique(test_nodes)

        mask_new = torch.zeros(data.x.size(0))
        mask_old = torch.ones(data.x.size(0))
        nb_new_nodes = 0

        for node in test_nodes:
            if node not in tr_nodes:
                nb_new_nodes+=1
            else:
                mask_new[node] = 1
            #print("new_nodes:",nb_new_nodes)
        mask_old = mask_old - mask_new
    try:
        embeds = torch.load(f"n2v_embeds_{dataset}.pth")
    except:
        start_idx = 0
        embeds = []
        for i in range(len(data.eis)):
            #print(f"Learning N2V timestemp {i+1}...")
            #print(f"[{start_idx},{start_idx + i}]")

            ei = data.tr(start_idx + i)
            num_nodes = data.x.size(0)

            all_nodes = torch.arange(num_nodes)
            connected_nodes = ei.view(1,-1).unique()
            remaining_nodes = torch.tensor(list(set(all_nodes.numpy())
                            - set(connected_nodes.numpy())))
            dummy_edges = remaining_nodes.repeat(2,1)
            all_edges = torch.cat((ei,dummy_edges),dim=1)

            
            #print(ei.shape)
            model_n2v = Node2Vec(all_edges,
                                embedding_dim=32,
                                walk_length=50,
                                context_size=25,
                                walks_per_node=25,
                                num_negative_samples=1,
                                p=1.0,
                                q=1.0,
                                sparse=True,
                                )

            loader = model_n2v.loader(batch_size=128, shuffle=True)
            optimizer = torch.optim.SparseAdam(list(model_n2v.parameters()), lr=0.01)

            def train():
                model_n2v.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model_n2v.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)

            @torch.no_grad()
            def test():
                model_n2v.eval()
                z = model_n2v()
                acc = model_n2v.test(
                    train_z=z[data.train_mask],
                    train_y=data.y[data.train_mask],
                    test_z=z[data.test_mask],
                    test_y=data.y[data.test_mask],
                    max_iter=150,
                )
                return acc

            for epoch in range(1, 151):
                loss = train()
                #acc = test()
                model_n2v.eval()
                z=model_n2v()
                print(z.shape)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')#, Acc: {acc:.4f}')
            model_n2v.eval()
            z=model_n2v()
            embeds.append(z)
        embeds = torch.stack(embeds)
        print(embeds.shape)
        torch.save(embeds, f"n2v_embeds_{dataset}.pth")
    
    print("************")

    
    opt = Adam(model.parameters(), lr=lr)

    best = (0, None)
    no_improvement = 0

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        zs = None

        # Get embedding        
        print(type(mask_new))
        zs = model(data.x, data.eis, data.tr, mask_new, mask_old, embeds)[:end_tr]
        print(zs.shape)

        if not pred:
            p,n,z = g.link_detection(data, data.tr, zs, nratio=nratio)
            
        else:
            p,n,z = g.link_prediction(data, data.tr, zs, nratio=nratio)
        

        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item()
        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr, mask_new, mask_old, embeds)[:end_tr]

            if not pred:
                p,n,z = g.link_detection(data, data.va, zs)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) ),
                    end=''
                )

                avg = sscores[0] + sscores[1]

            else:
                #enclude_tr=true permet de générer des arêtes négatifs qui ne font partie ni des arêtes de train ni de valid
                #enclude_tr=False permet de générer des arêtes négatifs qui ne font partie des arête de valid seulement
                dp,dn,dz = g.link_prediction(data, data.va, zs, include_tr=False)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.new_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tPr  %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) ),
                    end=''
                )

                avg = (
                    dscores[0] + dscores[1] 
                )

            if avg > best[0]:
                print('*')
                best = (avg, deepcopy(model))
                no_improvement = 0

            # Log any epoch with no progress on val set; break after 
            # a certain number of epochs
            else:
                print()
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > 100:
                    no_improvement += 1
                if no_improvement == PATIENCE:
                    print("Early stopping...\n")
                    break
        

    model = best[1]
    with torch.no_grad():
        model.eval()
        wandb.init(
        # Set the project where this run will be logged
        project="Val_score_for_Colab", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{d}_{run}", 
        # Track hyperparameters and run metadata
        config={
        "lr": args.lr,
        "architecture": "EulerCGN",
        "dataset": d,
        "prediction": pred
        })
        # Inductive
        if not pred:
            zs = model(data.x, data.eis, data.tr, mask_new, mask_old, embeds)[end_tr-1:]
            #for t in range(1,zs.size(0)):
            #    zs_mask = zs[t,:,:] * new_mask.view(-1, 1)
            #    avg = torch.mean(zs_mask,dim=0)
            #    zs_new = avg.view(1, -1).expand(data.x.size(0), -1) * new_keep.view(-1, 1)
            #    zs[t,:,:] = zs_mask + zs_new

        # Transductive
        else:
            zs = model(data.x, data.eis, data.all, mask_new, mask_old, embeds)[end_tr-1:]
            #for t in range(0,zs.size(0)-1):
            #    zs_mask = zs[t,:,:] * mask_new.view(-1, 1)
            #    avg = torch.mean(zs_mask,dim=0)
            #    zs_new = avg.view(1, -1).expand(data.x.size(0),-1) * keep_old.view(-1, 1)
            #    zs[t,:,:] = zs_mask + zs_new
        
        if not pred:
            zs = zs[1:]
            p,n,z = g.link_detection(data, data.te, zs, start=end_tr)
            t, f = model.score_fn(p,n,z)
            sscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Static LP:  %s
                '''
            % fmt_score(sscores))
            wandb.log({"AUC": sscores[0], "AP": sscores[1], "new nodes": nb_new_nodes, "best_val_score": best[0]})
            wandb.finish()
            return {'auc': sscores[0], 'ap': sscores[1]}

        else:              
            p,n,z = g.link_prediction(data, data.all, zs, start=end_tr-1)
            t, f = model.score_fn(p,n,z)
            dscores = get_score(t, f)

            p,n,z = g.new_link_prediction(data, data.all, zs, start=end_tr-1)
            t, f = model.score_fn(p,n,z)
            nscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Dynamic LP:     %s 
                    Dynamic New LP: %s 
                ''' %
                (fmt_score(dscores),
                 fmt_score(nscores))
            )

            wandb.log({"AUC": dscores[0], "AP": dscores[1], "New_AUC": nscores[0], "New_AP": nscores[1], "best_val_score": best[0]})
            wandb.finish()
            return {
                'pred-auc': dscores[0],
                'pred-ap': dscores[1],
                'new-auc': nscores[0], 
                'new-ap': nscores[1],
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predict',
        action='store_true',
        help='Sets model to train on link prediction rather than detection'
    )
    parser.add_argument(
        '--lstm',
        action='store_true'
    )

    '''
    0.02 is default as it's the best overall, but for the DBLP dataset, 
    lower LR's (0.005 in the paper) work better for new pred tasks
    Optimal LRs are: 
        +---------+-------+-------+-------+
        | Dataset | Det   | Pred  | New   | 
        +---------+-------+-------+-------+
        | Enron   | 0.02  | 0.02  | 0.2   |
        +---------+-------+-------+-------+
        | FB      | 0.01  | 0.02  | 0.1   |
        +---------+-------+-------+-------+
        | DBLP    | 0.02  | 0.02  | 0.005 | 
        +---------+-------+-------+-------+
    '''
    parser.add_argument(
        '--lr',
        type=float,
        default=0.02
    )

    parser.add_argument(
        '--folder',
        type=str,
        default='/mnt/raid0_24TB/isaiah/code/TGCN/src/data'
    )

    args = parser.parse_args()
    outf = 'euler.txt' 

    for d in ['enron10','fb']:#['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d, folder=args.folder)
        model = EulerGCN(data.x.size(1), 32, 16, lstm=args.lstm)

        stats = [
            train(
                run,
                d,
                deepcopy(model), 
                data, 
                pred=args.predict, 
                lr=args.lr
            ) for run in range(NUM_TESTS)
        ]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('LR: %0.4f\n' % args.lr)
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()