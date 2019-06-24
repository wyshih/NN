import torch as t
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error


class  NumData(Dataset):
        def __init__(self, dim, num):
            super().__init__()
            self.device = 'cpu'#'cuda' if t.cuda.is_available() else 'cpu'
            self.dim = dim
            self.num = num
            self.w = t.rand(1, dim, device = self.device)*13*t.tensor([1 if t.rand(1,1, device = self.device) >= 0.5 else -1 for i in range(self.dim)], device = self.device, dtype = t.float)
            self.b = t.rand(1, 1, device = self.device)*10
            self.data = self.datagen()
        def __len__(self):
            return self.data['X'].shape[0]
        def __getitem__(self, idx):
            return {'X': self.data['X'][idx], 'Y':self.data['Y'][idx]}
        def datagen(self):
            x = t.rand(self.num, self.dim, device = self.device )*7.7
            y = x.mm(self.w.t()) + self.b   #you can add noise here     
            return {'X':x, 'Y':y}
        def RMSE(self, py):
            return t.sqrt(t.sum(t.pow(py-self.data['Y'], 2))/self.data['Y'].shape[0])
if __name__ == '__main__':        
    mp.set_start_method('spawn')    #if you have CUDA error (3): initialization error, then add this one
    in_d = 3
    num = 1000
    Data = NumData(3, num)
    TData = NumData(3, 300)
    #print(Data.w, Data.b, len(Data), Data[:10])
    #Data
    B_D = DataLoader(Data, batch_size = 100, num_workers = 16)

    lrmodel = nn.Linear(in_d, 1).to(Data.device)
    #lrmodel = nn.Sequential(nn.Linear(in_d, 2), nn.Linear(2, 1)).to(Data.device)
    lossf = nn.MSELoss()

    optimizer = torch.optim.SGD(lrmodel.parameters(), lr=5e-5)

    for epoch in range(5):
        optimizer.zero_grad()
        for d in B_D:
            
            output = lrmodel(d['X'])
            loss = lossf(output, d['Y'])
            loss.backward()
        optimizer.step()
        if epoch %5 == 0:
            print('epoch : {0}, loss : {1}'.format(epoch, loss))
    print(Data.w, Data.b)
    print(list(lrmodel.parameters()))
    print('RMSE by selfdefinition, ', TData.RMSE(lrmodel(TData.data['X'])))
    print('RMSE, by torch ', t.pow(lossf(lrmodel(TData.data['X']), TData.data['Y']),0.5))
    print('MSE by sklearn, ', mean_squared_error(lrmodel(TData.data['X']).cpu().detach().numpy(), TData.data['Y'].cpu().detach().numpy()))
