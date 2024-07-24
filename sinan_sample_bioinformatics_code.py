import lightning as pl
from pathlib import Path
import pandas as pd
import torch, yaml, h5py, warnings, gc
from lightning.pytorch.strategies import DDPStrategy
from mrmr import mrmr_classif
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger     # Comet Logger, Neptune Logger, TensorBoard Logger can be used with PL

'''
This work is mainly adopted form the lightning.ai (formerly Pytorch_lightning) [6] and some references listed below while taken help from ChatGPT multiple times.
'''

gc.set_threshold(0)

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

config = load_config('config.yml')

warnings.filterwarnings("ignore")    
torch.set_float32_matmul_precision('high')

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: Path = config['data']['dataset_dir'], 
                 batch_size: int = config['data']['batchsize'], 
                 num_workers: int = config['train']['num_workers'],
                 ):
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.persistance = False
        if num_workers > 0:
            self.persistance = True
        self.batch_size = batch_size
        self.train_dataset = None  # Initialize train_dataset attribute
        self.val_dataset = None    # Initialize val_dataset attribute
        self.test_dataset = None   # Initialize test_dataset attribute

    def prepare_data(self):

        dataset_path = self.data_dir
        dataset = pd.read_csv(dataset_path, delimiter=',')

        num_features = 10

        # Perform feature selection using mRMR
        selected_features = mrmr_classif(X=dataset.drop('class', axis=1), 
                                         y=dataset['class'], 
                                         K=num_features)

        selected_features = dataset[selected_features]

        # drop first column from df
        sequences = selected_features.values
        labels = dataset['class'].values

        # split the datast into train, validation, and test sets for torch
        train_size = int(0.8 * len(sequences))
        val_size = int(0.1 * len(sequences))
        test_size = len(sequences) - train_size - val_size

        self.train_sequences = sequences[:train_size]
        self.train_labels = labels[:train_size]

        self.val_sequences = sequences[train_size:train_size + val_size]
        self.val_labels = labels[train_size:train_size + val_size]

        self.test_sequences = sequences[train_size + val_size:]
        self.test_labels = labels[train_size + val_size:]

    def setup(self):
        # Create the TensorDataset using the loaded data
        self.train_dataset = torch.utils.data.TensorDataset(self.train_sequences, 
                                                            self.train_labels).float()
        self.val_dataset = torch.utils.data.TensorDataset(self.val_sequences, 
                                                        self.val_labels).float()
        self.test_dataset = torch.utils.data.TensorDataset(self.test_sequences, 
                                                        self.test_labels).float()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                          persistent_workers = self.persistance,
                          drop_last = True,
                          pin_memory = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                          persistent_workers = self.persistance,    # [3]
                          pin_memory = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, 
                          persistent_workers = self.persistance,    # [3]
                          pin_memory = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)


class TransformerBlock(torch.nn.Module):
    def __init__(self, embeds_size):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=embeds_size, num_heads=8)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embeds_size, 4 * embeds_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embeds_size, embeds_size),
        )
        self.ln1 = torch.nn.LayerNorm(embeds_size)
        self.ln2 = torch.nn.LayerNorm(embeds_size)

    def forward(self, x):
        attended = self.attention(x, x, x)[0]
        x = self.ln1(x + attended)
        fed_forward = self.feed_forward(x)
        x = self.ln2(x + fed_forward)
        return x

# Define the modified transformer model
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, embeds_size, block_size, num_classes, drop_prob):
        super(self).__init__()
        self.tok_embs = torch.nn.Embedding(vocab_size, embeds_size)
        self.pos_embs = torch.nn.Embedding(block_size, embeds_size)
        self.block = TransformerBlock(embeds_size)
        self.ln1 = torch.nn.LayerNorm(embeds_size)
        self.ln2 = torch.nn.LayerNorm(embeds_size)
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(embeds_size, embeds_size),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(drop_prob),
            torch.nn.Linear(embeds_size, embeds_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embeds_size, num_classes),
            torch.nn.Softmax(dim=1),
        )
        print("Number of parameters: %.2fM" % (self.num_params() / 1e6,))

    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, seq):
        B, T = seq.shape
        seq = seq.long()  # Convert seq tensor to Long type
        device = seq.device
        embedded = self.tok_embs(seq)
        embedded = embedded + self.pos_embs(torch.arange(T, device=device))
        output = self.block(embedded)
        output = output.mean(dim=1)
        output = self.classifier_head(output)
        return output

class Lit_virus_classifier(pl.LightningModule):
    def __init__(self, Model):
        super().__init__()
        
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        # call models
        self.model = Model
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def update(self, seq, labels):
        pred_seq = self.forward(seq) 
            
        cross_entropy_loss = self.cross_entropy(pred_seq, labels)
            
        return cross_entropy_loss
    
    def training_step(self, batch, batch_idx):
        seq, label = batch
        device = seq.device
        pred_seq,  pred_labels= seq.to(device), label.to(device)
        
        cross_entropy_loss = self.update(seq, label)
        
        self.log_dict({
            'classification_loss': cross_entropy_loss, 
            },
            on_step=False,      # on_step means that to show detail of every epoch separately if it's ON. 
            on_epoch=True, 
            prog_bar=True, 
            logger=True)
        self.logger.experiment.add_histogram("classification_loss", cross_entropy_loss, global_step=trainer.global_step)
        
        return cross_entropy_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_pred_label = self.forward(x)
        val_loss = self.mse_metric(val_pred_label, y)
        self.logger.experiment.add_histogram("Validation: loss", val_loss, global_step=trainer.global_step)
        self.log('val-loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   
    
    def test_step(self, batch, batch_idx):
        seq, label = batch
        val_pred_label = self.forward(seq)
        test_loss = self.mse_metric(val_pred_label, label)
        self.log('test-loss', self.test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram("Testing: loss", test_loss, global_step=trainer.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, label = batch
        device = seq.device
        seq, label = seq.to(device), label.to(device)
        output_label = self.forward(seq)
        return output_label
    
    def _common_step(self, batch, batch_idx):
        # Unpack the batch
        seq, label = batch
        # Forward pass to get model predictions
        output_label = self.forward(seq)
        # Compute the loss using the provided metric
        loss = self.mse_metric(output_label, label)
        return loss
    
    def visualization(self):
        return
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config['train']['base_lr'])
    
if __name__ == '__main__':
    custom_profiler = pl.pytorch.profilers.PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler('tb_logs/virusClassifier'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule = torch.profiler.schedule(
                                           wait=1, 
                                           warmup=1, 
                                           active=3, 
                                           repeat=1
                                           )
    )
    
    logger = TensorBoardLogger('tb_logs', name="virus_classifier_model_logger")
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath=config['model']['checkpoint_dir'], 
                                                                filename = config['model']['name'],
                                                                save_top_k=2,           
                                                                mode="min",
                                                                monitor="val-loss")
    # Initialize a PyTorch SummaryWriter
    writer = SummaryWriter(logger.log_dir)
    strategy = DDPStrategy(static_graph=True)   # [4]
    trainer = pl.pytorch.Trainer(
                    accelerator=config['train']['accelerator'], 
                    devices = config['train']['device'], 
                    max_epochs=config['train']['max_epochs'], 
                    precision=config['train']['precision'], 
                    enable_checkpointing=True,
                    check_val_every_n_epoch=1,
                    detect_anomaly=True,
                    # gradient_clip_val=0.5,
                    # gradient_clip_algorithm="value",
                    enable_progress_bar=True,
                    # enable_model_summary=True, # summary: will show every detail of training after finishing of epochs, such as time etc 
                    profiler=custom_profiler,
                    # deterministic=True,
                    logger=logger,
                    callbacks= checkpoint_callback,
                    )
    
    with trainer.init_module():    # [3]
        datamodule = DataModule()
        datamodule.prepare_data()
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        # test_loader = datamodule.test_dataloader()
        pytorch_model = Transformer(
                    vocab_size=config['model']['vocab_size'], 
                    embeds_size=config['model']['embeds_size'], 
                    block_size=config['model']['block_size'], 
                    num_classes=config['model']['num_classes'], 
                    drop_prob=config['model']['drop_prob'],
                    )
        VirusClassifier = Lit_virus_classifier(pytorch_model)
    
    """
    to train multiple models, [1] can be used
    
    """
    
    trainer.fit(model=VirusClassifier, train_dataloaders=train_loader)
    trainer.validate(model=VirusClassifier, dataloaders=val_loader)
    
    # Iterate over the model parameters and add histograms to TensorBoard
    for name, param in pytorch_model.named_parameters():
        writer.add_histogram(name, param, global_step=trainer.global_step)
    # Close the SummaryWriter
    writer.close()

""" References
[1] https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
[2] https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
[3] https://github.com/Lightning-AI/pytorch-lightning/issues/19373
[4] https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#ddpstrategy
[5] https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html#lightning.pytorch.callbacks.BatchSizeFinder
[6] https://lightning.ai/blog/pl-tutorial-and-overview/
"""