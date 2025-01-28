import pytorch_lightning as pl
import torch.nn as nn
from basic_layers import *
from fourier_layers import *
from torch import optim
class FNO(pl.LightningModule):
    def __init__(self,     
                    wavenumber, features_, 
                    padding = 9, 
                    lifting = None, 
                    proj =  None, 
                    dim_input = 1, 
                    with_grid= True,
                    add_term = True, 
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    eta_min = 5e-4):
        super(FNO, self).__init__()
        self.with_grid = with_grid
        self.padding = padding   
        self.layers = len(wavenumber)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.add_term = add_term
        self.criterion = nn.MSELoss()
        self.criterion_val = nn.MSELoss()
        if with_grid == True: 
            dim_input+=4 
        self.lifting = FC_nn([dim_input, features_//2, features_], 
                                outermost_norm=False
                                )
         
        self.proj =  FC_nn([features_, features_//2, 2], 
                            outermost_norm=False
                                )
        self.fno = []
        for l in range(self.layers-1):
            self.fno.append(FourierLayer(features_ = features_, 
                                        wavenumber=[wavenumber[l]]*2))
        self.fno.append(FourierLayer(features_=features_, 
                                        wavenumber=[wavenumber[-1]]*2, 
                                        is_last= True))
        self.fno =nn.Sequential(*self.fno)
    def forward(self, sos, src):
        # forward process of FNO
        #x = torch.cat((sos, src), dim=-1)
        x = sos
        if self.with_grid == True:
          grid = get_grid2D(x.shape, x.device)
          x = torch.cat((x,src, grid), dim=-1)
        x = self.lifting(x)  
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.pad(x, [0,self.padding, 0,self.padding]) 
        x = self.fno(x)
        x = x[..., :-self.padding, :-self.padding] 
        x = x.permute(0, 2, 3, 1 )
        x =self.proj(x)  
        if self.add_term == True:
            x = torch.view_as_real(torch.view_as_complex(src.to(x.device))*(1+torch.view_as_complex(x)))
        return x

    def training_step(self, batch: torch.Tensor, batch_idx):
        # One step training 
        sos,src,y = batch
        batch_size = sos.shape[0]
        out = self(sos,src)
        loss = self.criterion(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        # One step validation
        sos,src,y= val_batch
        batch_size = sos.shape[0]
        out = self(sos,src)
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
        
    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if  scheduler is None:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
        return {
        "optimizer": optimizer,
        "lr_scheduler": { 
            "scheduler": scheduler
        },
        }